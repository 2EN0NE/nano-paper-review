// lib.js 纯函数的单元测试
// 运行: node --test test_lib.js

const test = require("node:test");
const assert = require("node:assert/strict");
const lib = require("./lib");

test("sanitizeFilename: 清洗文件系统非法字符", () => {
	assert.equal(lib.sanitizeFilename('A/B:C*D?E"F<G>H|I'), "A_B_C_D_E_F_G_H_I");
});

test("sanitizeFilename: 压缩连续空白与下划线", () => {
	assert.equal(lib.sanitizeFilename("  hello   world  "), "hello world");
	assert.equal(lib.sanitizeFilename("a___b"), "a_b");
});

test("sanitizeFilename: 去除首尾点并截断", () => {
	assert.equal(lib.sanitizeFilename("...dot..."), "dot");
	assert.equal(lib.sanitizeFilename("x".repeat(200)).length, 150);
});

test("extractArticleUrls: 提取并去重，保持出现顺序", () => {
	const js =
		'a="/2020/01/01/foo.html" b="/2020/01/01/foo.html" c="/2021/02/02/bar.html"';
	assert.deepEqual(lib.extractArticleUrls(js), [
		"/2020/01/01/foo.html",
		"/2021/02/02/bar.html",
	]);
});

test("extractArticleUrls: 无匹配返回空数组", () => {
	assert.deepEqual(lib.extractArticleUrls("no urls here"), []);
});

test("filterByYear: 按年过滤并排序", () => {
	const urls = [
		"/2021/02/02/b.html",
		"/2021/01/01/a.html",
		"/2020/01/01/c.html",
	];
	assert.deepEqual(lib.filterByYear(urls, "2021"), [
		"/2021/01/01/a.html",
		"/2021/02/02/b.html",
	]);
});

test("extractTitle: 去掉站点后缀", () => {
	assert.equal(lib.extractTitle("标题 | 美团 · 技术团队"), "标题");
	assert.equal(lib.extractTitle("无分隔符标题"), "无分隔符标题");
	assert.equal(lib.extractTitle(""), "");
});

test("buildManifestRow + parseManifest: 往返一致", () => {
	const rec = {
		year: "2020",
		date: "01-01",
		title: "测试 标题",
		slug: "test",
		url: "/2020/01/01/test.html",
		status: "OK",
		filename: "01-01_测试_标题.pdf",
		sizeBytes: 123,
	};
	const text = lib.MANIFEST_HEADER + lib.buildManifestRow(rec) + "\n";
	const parsed = lib.parseManifest(text);
	assert.equal(parsed.size, 1);
	const got = parsed.get("/2020/01/01/test.html");
	assert.equal(got.status, "OK");
	assert.equal(got.title, "测试 标题");
	assert.equal(got.sizeBytes, "123");
});

test("buildManifestRow: 转义制表符与换行", () => {
	const rec = {
		year: "2020",
		date: "01-01",
		title: "a\tb\nc",
		slug: "s",
		url: "/u",
		status: "OK",
		filename: "f.pdf",
		sizeBytes: 0,
	};
	const row = lib.buildManifestRow(rec);
	assert.ok(!row.includes("\n"));
	assert.ok(!row.includes("\r"));
	assert.equal(row.split("\t").length, 8);
	assert.equal(row.split("\t")[2], "a b c");
});

test("parseManifest: 跳过表头与空行、忽略不完整行", () => {
	const text = lib.MANIFEST_HEADER + "\n" + "bad line only\n";
	const parsed = lib.parseManifest(text);
	assert.equal(parsed.size, 0);
});
