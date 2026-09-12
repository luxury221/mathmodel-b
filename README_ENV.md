# 2026 数学建模 B 题：本地实验环境

Python 环境准备日期：2026-09-10；模拟器安装与启动验证日期：2026-09-11。项目目录：`D:\数模B题\B题`。

## 当前状态

- Python 实验环境已安装，解释器、依赖、缓存、临时文件及输出目录均配置在 D 盘。
- Python 版本固定为 **3.12.14**，环境管理工具为 **uv 0.12.12**。
- 36 项直接依赖，共 152 个 Python 包，版本和 SHA-256 哈希保存在 `requirements.lock.txt`。
- 数值计算、几何操作、优化求解器、中文绘图、文档读写、本地 HTTP/JSON 和 Jupyter 内核自检已通过。
- **官方模拟器已安装并通过本机启动验证**：使用用户下载的 `Jammers-simulator-win64.7z` 精简版，成功加载本机已有的 WebView2 `152.0.4191.66`。本机不需要再补下载完整包。
- 主程序窗口正常创建，2026 端口只保留在本机回环地址；应用数据和 WebView2 缓存均写入 D 盘。验证进程已正常关闭，没有留下后台实例。
- 未注册或登录竞赛账号，未启动演练或正式测试，未编写任何本题搜索、定位、路径规划或清除算法。

自检状态以 `reports/environment.json` 为准；Jupyter 网页服务检查以 `reports/jupyter-server-check.json` 为准；模拟器本机启动证据见 `reports/simulator-startup-check.json`。尚未验证登录页面内容、账号登录、在线连通性或官方演练流程。自检输出不是赛题实验结果。

## 一键入口

| 文件 | 用途 |
| --- | --- |
| `Open_Experiment_Terminal.cmd` | 打开已加载 D 盘环境变量的 PowerShell 终端 |
| `Start_Jupyter.cmd` | 启动 JupyterLab，只监听本机，保留默认登录令牌保护 |
| `Start_Simulator.cmd` | 启动已安装的官方模拟器，固定使用 D 盘工作目录和缓存 |
| `Check_Environment.cmd` | 重新检查依赖和主要功能，生成检查报告 |

Jupyter 启动后，使用终端打印的带令牌本地地址访问。使用结束后在该终端按 `Ctrl+C` 关闭。不要将带令牌地址发给他人。

也可以在项目目录运行：

```powershell
. .\scripts\env.ps1
python --version
python -c "import sys; print(sys.executable)"
.\scripts\start-jupyter.ps1
```

Python 路径应为 `D:\数模B题\B题\.venv\Scripts\python.exe`。不要直接使用系统 `python` 或现有 Anaconda 代替。

仅执行 `.venv\Scripts\Activate.ps1` 不足以设置全部缓存目录；应使用 `scripts/env.ps1` 或一键入口。

## 已安装的工具

| 类别 | 主要软件包 |
| --- | --- |
| 数值与数据处理 | NumPy、SciPy、pandas、SymPy、Numba、joblib |
| 平面几何与图结构 | Shapely/GEOS、NetworkX |
| 优化与运筹 | SciPy 优化、CVXPY 及其可用求解器、OR-Tools |
| 常见统计与机器学习 | scikit-learn |
| 可视化 | Matplotlib、Seaborn、Plotly、Pillow，已检查微软雅黑中文显示 |
| HTTP、配置与日志辅助 | requests、httpx、tenacity、Pydantic、PyYAML、Rich、tqdm、psutil |
| 交互实验 | JupyterLab、ipykernel、ipywidgets、nbformat、nbconvert |
| Excel、Word、PDF | openpyxl、python-docx、pypdf、pdfplumber、PyMuPDF、ReportLab |
| 开发检查 | Ruff、pytest |

已经存在的 VS Code、7-Zip 和 MiKTeX 不做全局升级。此次实验基线是 Python；未安装或激活 MATLAB，也没有假定后续算法需要 GPU、深度学习框架或商业求解器。

## 官方模拟器安装与使用

官方来源是 `附件\附件1.docx` 中的百度网盘链接，提取码为 `2026`：

```text
https://pan.baidu.com/s/1P1yfVjY0RufU93XOdzhOLw?pwd=2026
```

2026-09-10 读取到的官方目录包含：

| 文件 | 服务器列出的大小 |
| --- | ---: |
| `Jammers-simulator-full-win64.7z` | 236,850,965 字节 |
| `Jammers-simulator-win64.7z` | 6,109,513 字节 |
| `先读下载说明.pdf` | 247,476 字节 |
| `模拟器操作演示.mp4` | 130,689,914 字节 |

2026-09-11 已收到并安装精简版，压缩包大小与上述列表一致，7-Zip 完整性检查通过。包内主程序约 18.8 MB，本机现有 WebView2 可以正常加载，无须另外安装 .NET 或重复下载完整包来满足这次启动验证。

安装位置：`D:\数模B题\B题\simulator\official\Jammers-simulator\jammers-simulator.exe`。

归档 SHA-256：`23cd362c256620b61d5be050655a6cc41a3868b1099231452947c1741f88f086`。

完整包的内部文件没有检查。已安装主程序的 Authenticode 状态为 `NotSigned`；本次没有关闭 Windows 安全防护或更改系统信任策略。

以下导入命令仅供在空安装目录中复现，**本机当前不必再次执行**：

1. 在正常浏览器或百度网盘客户端下载官方压缩包，保留原文件名。
2. 保存到 `D:\数模B题\B题\.runtime\downloads`，不要放到空间紧张的 C 盘。
3. 在实验终端执行：

```powershell
.\scripts\install-simulator.ps1 -ArchivePath '.\.runtime\downloads\Jammers-simulator-win64.7z'
```

导入脚本使用已有的 `D:\7-Zip\7z.exe`，先检查归档路径安全性和文件完整性，再解压到 `simulator\official`。它不覆盖已有模拟器数据，不自动运行可执行文件，不登录账号，也不发起测试。若无法唯一识别主程序，会保留解压结果并要求人工确认。

成功导入后会生成 `simulator\installation.json`，记录来源、归档 SHA-256 和主程序路径。本地 SHA-256 是完整性记录，不是组委会提供的数字签名或真实性认证。

日常请双击 `Start_Simulator.cmd`，或在实验终端交互启动：

```powershell
.\scripts\start-simulator.ps1
```

首次登录所需参赛队号、队员信息和密码，请本人在官方模拟器中输入，不要写进代码、聊天记录或支撑材料。模拟器运行数据与可执行文件应保存在同一个 D 盘目录中；后续迁移时不要只移动 exe 文件。

必须使用项目启动器，以保留 D 盘缓存配置；不建议直接双击 exe。启动器会拒绝重复启动同一路径的实例，避免端口和数据冲突。

实际观察到的目录：

- 模拟器数据：`simulator\official\Jammers-simulator\JammersSimulatorData`。
- WebView2 用户数据：`.cache\simulator\roaming\jammers-simulator.exe\EBWebView`。
- 模拟器进程专用的 Local/Roaming 配置：`.cache\simulator\local`、`.cache\simulator\roaming`。

初始化生成的 SQLite 队列文件是模拟器自己的运行数据，不能把它们当作正式测试日志。不要手动修改或删除这些数据文件。

此前自动下载失败的历史诊断保留在 `reports/simulator_download.json`；该下载问题已通过用户提供的官方压缩包解决，不再是安装阻碍。

## 与模拟器交互的边界

- 默认接口为 `http://127.0.0.1:2026`，只有模拟器开始某次测试并完成倒计时后才开放。
- 四类动作是 `/enter`、`/measure`、`/clear`、`/exit`。此次环境自检未调用这些接口，HTTP 自检使用随机本地端口。
- `config/experiment.example.json` 仅是未来执行器的配置模板；目前没有程序读取它。`allow_formal_tests=false` 不是模拟器自身的保护开关，不能阻止人在模拟器界面点击正式测试。
- 正式测试次数有限；不应将正式测试作为环境探测或网络连通性检查。
- 演练与正式测试的程序和算法均留待后续指令，现有检查程序只验证软件安装。
- 各次正式日志应保持官方原文件名，不修改、不伪造；目前日志目录为空。

## 目录与 D 盘隔离

| 路径 | 内容 |
| --- | --- |
| `.runtime\python` | 项目独立的 CPython 运行时 |
| `.runtime\uv` | 经官方发布校验和核对的 uv 可执行文件 |
| `.runtime\downloads` | 安装包及其来源信息 |
| `.venv` | 项目 Python 包与 Jupyter 内核注册 |
| `.cache` | 下载、临时文件、Python 字节码、Matplotlib、Numba 等缓存 |
| `.jupyter`、`.ipython` | 项目专用交互环境配置和运行数据 |
| `simulator` | 已安装的官方模拟器及其运行数据 |
| `src`、`notebooks` | 后续算法代码及实验笔记，当前留空 |
| `logs\practice`、`logs\formal` | 后续测试日志，当前留空 |
| `outputs` | 后续实验图表和结果，当前留空 |
| `reports` | 软件环境自检报告及明确标记的诊断样例 |

没有修改全局 PATH、系统临时目录、已有 Anaconda 环境或 Windows 执行策略。启动器中的 `ExecutionPolicy Bypass` 仅影响它启动的当前 PowerShell 进程。

VS Code 工作区已指定本项目解释器和终端入口。Jupyter 中请选择 **Python (B2026, D drive)** 内核；该内核也固定携带 D 盘缓存环境变量。

## 复现与检查

```powershell
.\scripts\setup.ps1
.\scripts\check-environment.ps1
```

`setup.ps1` 使用 `.python-version` 和带哈希的 `requirements.lock.txt` 复现依赖，只作用于本项目环境。比赛期间不要无目的升级依赖；如确需更改，再有意识地使用 `-RefreshLock` 并重新检查。

功能检查覆盖：直接依赖导入、线性代数、符号计算、Numba 本地编译、GEOS 几何运算、CVXPY/OR-Tools 求解器、中文 PNG/PDF 导出、Excel/Word/PDF 读写、本机 HTTP/JSON，以及真实启动 Jupyter 内核执行一段环境诊断代码。

自检脚本不评估算法优劣，不提供清除率、定位精度、路径时间等竞赛成绩，也不代替官方模拟器验收。

## 安装来源

- uv 官方发布：`https://github.com/astral-sh/uv/releases`。
- uv 官方安装说明：`https://docs.astral.sh/uv/getting-started/installation/`。
- Python 运行时由 uv 管理并安装到项目目录。
- Python 包由 PyPI 解析，精确版本和哈希在依赖锁文件中。
- 模拟器仅使用本地官方附件提供的分享链接；未下载非官方替代品。

uv 压缩包来源与 SHA-256 记录保存在 `.runtime/downloads/uv-source.json`。
