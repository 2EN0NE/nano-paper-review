# 美团技术博客 PDF 逐年下载工具

将美团技术团队博客（<https://tech.meituan.com）的文章按年份下载为> PDF，每年一个文件夹，每篇文章一个 PDF。

## 原理

1. 美团博客是 VuePress SPA，全站文章列表硬编码在 `assets/app-*.js` 的路由映射中（约 640 篇）。
2. 每篇文章正文（markdown 编译后的 HTML）硬编码在各自页面的 JS bundle 里。
3. 用 Playwright + Chromium headless 渲染完整页面 → `page.pdf()`，注入打印 CSS 只保留正文。

## 目录结构

```
download_meituan_tech_pdf/
├── lib.js             # 纯函数模块（URL 提取 / 文件名清洗 / manifest 解析）
├── download_year.js   # 主脚本：node download_year.js <年份>
├── test_lib.js        # lib.js 单元测试（node:test）
├── package.json       # playwright 依赖
└── README.md
```

输出（项目根目录）：

```
meituan-blog-pdf/          # 已加入 .gitignore
├── 2020/                  # 每年一个文件夹
│   ├── 01-09_标题.pdf      # 文件名：MM-DD_文章标题.pdf
│   └── ...
├── manifest.csv           # 全量清单：year/date/title/slug/url/status/filename/size
└── download_YYYY.log      # 每次运行日志
```

## 安装

```bash
cd scripts/download_meituan_tech_pdf
npm install          # 安装 playwright

# 首次需下载 Chromium（国内网络建议用镜像，约 280MB，仅一次）
PLAYWRIGHT_DOWNLOAD_HOST=https://cdn.npmmirror.com/binaries/playwright npx playwright install chromium
```

浏览器缓存位于 `~/Library/Caches/ms-playwright/`（macOS，用户级），不受本目录清理影响。

## 使用

```bash
node download_year.js 2021          # 下载 2021 全年
node download_year.js 2021 3        # 只下载前 3 篇（小批次验证用）
```

特性：

- **断点续传**：`manifest.csv` 中 `status=OK` 的文章自动跳过，可安全中断后重跑。
- **失败重试**：每篇最多 2 次。
- **限速**：每篇之间延迟 800ms，避免压服务器。
- **完整性核对**：`manifest.csv` 记录每篇的标题/URL/状态/大小，可据此核对有无遗漏。

## 测试

```bash
node --test test_lib.js     # 或 npm test
```

## 年份范围

| 年份 | 文章数 |
|---|---|
| 2013 | 4 |
| 2014 | 38 |
| 2015 | 24 |
| 2016 | 51 |
| 2017 | 77 |
| 2018 | 107 |
| 2019 | 49 |
| 2020 | 56 |
| 2021 | 54 |
| 2022 | 63 |
| **2013–2022 合计** | **523** |
