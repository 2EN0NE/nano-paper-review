// 美团技术博客 PDF 下载主脚本
// 用法: node download_year.js <年份> [limit]
//   limit 可选：只下载前 N 篇（测试用），省略则下载全部
// 断点续传：manifest.csv 中 status=OK 的文章自动跳过

const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");
const {
	extractArticleUrls,
	filterByYear,
	sanitizeFilename,
	extractTitle,
	parseManifest,
	buildManifestRow,
	MANIFEST_HEADER,
} = require("./lib");

const YEAR = process.argv[2];
const LIMIT = process.argv[3] ? parseInt(process.argv[3], 10) : 0;
if (!YEAR || !/^\d{4}$/.test(YEAR)) {
	console.error("用法: node download_year.js <年份> [limit]");
	process.exit(1);
}

const PROJECT_ROOT = path.resolve(__dirname, "../..");
const OUT_ROOT = path.join(PROJECT_ROOT, "meituan-blog-pdf");
const OUT_DIR = path.join(OUT_ROOT, YEAR);
const MANIFEST = path.join(OUT_ROOT, "manifest.csv");
const LOG = path.join(OUT_ROOT, `download_${YEAR}.log`);

fs.mkdirSync(OUT_DIR, { recursive: true });

// ---------- 日志 ----------
const logStream = fs.createWriteStream(LOG, { flags: "a" });
function log(msg) {
	const line = `[${new Date().toISOString()}] ${msg}`;
	console.log(line);
	logStream.write(line + "\n");
}

const PRINT_CSS = `
  @media print {
    .vp-navbar, .vp-sidebar, .vp-footer, .vp-toggle-color-mode-button,
    .page-nav, .vp-toc, .theme-doc-footer, .vp-breadcrumb,
    .vp-home, .vp-home-content, .post-list, .vp-home-carousel { display: none !important; }
    .vp-page, .theme-default-content, .theme-container { width: 100% !important; max-width: 100% !important; }
    body { background: #fff !important; }
    html[data-theme="dark"] body { background: #fff !important; }
    .theme-default-content img { max-width: 100% !important; }
    pre, code { white-space: pre-wrap !important; word-break: break-all !important; }
  }
`;

// ---------- 获取文章列表 ----------
async function getArticleUrls(year) {
	const home = await fetch("https://tech.meituan.com/").then((r) => r.text());
	const m = home.match(/src="\/(assets\/app-[^"]+\.js)"/);
	if (!m) throw new Error("首页找不到 app bundle 引用");
	const appUrl = "https://tech.meituan.com/" + m[1];
	log(`app bundle: ${appUrl}`);
	const app = await fetch(appUrl).then((r) => r.text());
	const urls = extractArticleUrls(app);
	const yearUrls = filterByYear(urls, year);
	log(`全站 ${urls.length} 篇，${year} 年 ${yearUrls.length} 篇`);
	return yearUrls;
}

// ---------- manifest 读取（断点续传） ----------
function loadDone() {
	if (!fs.existsSync(MANIFEST)) return new Map();
	const map = parseManifest(fs.readFileSync(MANIFEST, "utf8"));
	const done = new Set();
	for (const [url, rec] of map) {
		if (rec.status === "OK") done.add(url);
	}
	return done;
}

function appendManifest(row) {
	if (!fs.existsSync(MANIFEST)) fs.writeFileSync(MANIFEST, MANIFEST_HEADER);
	fs.appendFileSync(MANIFEST, row + "\n");
}

// ---------- 主流程 ----------
(async () => {
	const urls = await getArticleUrls(YEAR);
	const done = loadDone();
	let todo = urls.filter((u) => !done.has(u));
	if (LIMIT > 0) todo = todo.slice(0, LIMIT);
	log(`已完成 ${urls.length - todo.length} 篇，待下载 ${todo.length} 篇`);

	const browser = await chromium.launch({ headless: true });
	let ok = 0;
	let fail = 0;

	for (let i = 0; i < todo.length; i++) {
		const url = todo[i];
		const mm = url.match(/\/(\d{4})\/(\d{2})\/(\d{2})\/([A-Za-z0-9_-]+)\.html/);
		const [, y, mo, d, slug] = mm;
		const date = `${mo}-${d}`;
		let title = slug;
		let status = "FAIL";
		let filename = "";

		for (let attempt = 1; attempt <= 2; attempt++) {
			const page = await browser.newPage({
				viewport: { width: 1280, height: 900 },
				colorScheme: "light",
			});
			try {
				await page.addStyleTag({ content: PRINT_CSS });
				await page.goto("https://tech.meituan.com" + url, {
					waitUntil: "domcontentloaded",
					timeout: 60000,
				});
				await page
					.waitForSelector(".theme-default-content", { timeout: 30000 })
					.catch(() => {});
				await page
					.waitForLoadState("networkidle", { timeout: 25000 })
					.catch(() => {});
				await page.waitForTimeout(1200);

				const fullTitle = await page.title();
				title = extractTitle(fullTitle) || slug;
				filename = `${date}_${sanitizeFilename(title)}.pdf`;

				await page.pdf({
					path: path.join(OUT_DIR, filename),
					format: "A4",
					printBackground: true,
					margin: { top: "18mm", bottom: "18mm", left: "16mm", right: "16mm" },
				});
				status = "OK";
			} catch (e) {
				log(`  重试 #${attempt} ${url} 失败: ${e.message.split("\n")[0]}`);
			} finally {
				await page.close().catch(() => {});
			}
			if (status === "OK") break;
		}

		let sizeBytes = 0;
		if (status === "OK" && filename) {
			try {
				sizeBytes = fs.statSync(path.join(OUT_DIR, filename)).size;
			} catch {}
			ok++;
		} else {
			fail++;
		}

		appendManifest(
			buildManifestRow({
				year: y,
				date,
				title,
				slug,
				url,
				status,
				filename,
				sizeBytes,
			}),
		);
		log(
			`[${i + 1}/${todo.length}] ${status} ${title.slice(0, 40)} (${(sizeBytes / 1024 / 1024).toFixed(2)}MB)`,
		);

		// 限速：每篇之间延迟
		await new Promise((r) => setTimeout(r, 800));
	}

	await browser.close();
	log(`===== 完成：成功 ${ok}，失败 ${fail}，共 ${todo.length} =====`);
	logStream.end();
})().catch((e) => {
	log("FATAL: " + e.message);
	logStream.end();
	process.exit(1);
});
