// 美团技术博客 PDF 下载工具 —— 纯函数模块（可测试）

// 从 VuePress app bundle 文本中提取全部文章 URL（去重）
// URL 形如 /YYYY/MM/DD/slug.html
function extractArticleUrls(appJsText) {
	const urls =
		String(appJsText || "").match(
			/\/\d{4}\/\d{2}\/\d{2}\/[A-Za-z0-9_-]+\.html/g,
		) || [];
	return [...new Set(urls)];
}

// 按年份过滤并排序
function filterByYear(urls, year) {
	const prefix = "/" + year + "/";
	return (urls || []).filter((u) => u.startsWith(prefix)).sort();
}

// 文件名清洗：移除文件系统非法字符，压缩空白，截断
function sanitizeFilename(name) {
	return String(name == null ? "" : name)
		.replace(/[\\/:*?"<>|\u0000-\u001f]/g, "_")
		.replace(/\s+/g, " ")
		.replace(/_+/g, "_")
		.trim()
		.replace(/^\.+|\.+$/g, "")
		.slice(0, 150);
}

// 从页面 <title> 提取文章标题（去掉站点后缀 " | 美团 · 技术团队"）
function extractTitle(pageTitle) {
	return (pageTitle || "").split(" | ")[0].trim();
}

const MANIFEST_HEADER =
	"year\tdate\ttitle\tslug\turl\tstatus\tfilename\tsize_bytes\n";

// 解析 manifest 文本 → Map<url, record>
function parseManifest(text) {
	const map = new Map();
	if (!text) return map;
	const lines = text.split("\n").slice(1); // 跳过表头
	for (const line of lines) {
		if (!line.trim()) continue;
		const parts = line.split("\t");
		if (parts.length < 8) continue;
		const [year, date, title, slug, url, status, filename, sizeBytes] = parts;
		map.set(url, { year, date, title, slug, url, status, filename, sizeBytes });
	}
	return map;
}

// 构建 manifest 数据行（转义制表符/换行）
function buildManifestRow(record) {
	const esc = (s) => String(s == null ? "" : s).replace(/[\t\n\r]/g, " ");
	return [
		esc(record.year),
		esc(record.date),
		esc(record.title),
		esc(record.slug),
		esc(record.url),
		esc(record.status),
		esc(record.filename),
		record.sizeBytes || 0,
	].join("\t");
}

module.exports = {
	extractArticleUrls,
	filterByYear,
	sanitizeFilename,
	extractTitle,
	parseManifest,
	buildManifestRow,
	MANIFEST_HEADER,
};
