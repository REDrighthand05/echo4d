# WO-gs-ply-reader：高斯 PLY 顶点表读取器（DS Flash 工单）

## 背景
echo_super 超级集成新增内核：3DGS 高斯泼溅 .ply → 彩色点云适配器。
本工单只写其中一个边界明确的小函数：PLY 文件顶点表读取。颜色转换、
不透明度筛选、EchoObject 适配由 Muse 写，不在本工单范围。

## 交付物
单个自包含 Python 函数（只许 import numpy 与标准库）：

```python
def read_ply_vertices(path) -> dict[str, "np.ndarray"]:
    ...
```

## 契约
- 输入：PLY 文件路径（str 或 PathLike）。
- 解析 PLY 1.0 头：`format ascii 1.0` 与 `format binary_little_endian 1.0` 两种都要支持；
  `binary_big_endian` 抛 ValueError。
- 头里逐个 element 顺序读取；只返回名为 `vertex` 的 element：
  dict，键=属性名，值=一维 numpy 数组（长度=顶点数，dtype 与文件声明一致：
  float→float32，double→float64，uchar→uint8，int→int32 等，含 char/short/ushort/uint 别名）。
- 其它 element（face 等）要正确跳过（包括其 list 属性，按 count 类型+元素类型计算字节数）。
- vertex element 里若出现 list 属性，抛 ValueError（高斯文件不会有，属损坏/不支持）。
- 头损坏、缺 end_header、文件截断：抛 ValueError，信息里带一句原因。
- 不打印任何东西；不写文件；纯函数。
- 代码带简短 docstring 说明契约即可，不要长注释。

## 验收（Muse 台架）
- Muse 手工构造的合成高斯 PLY（binary_little_endian，标准 3DGS 属性布局
  x,y,z,nx,ny,nz,f_dc_0..2,opacity,scale_0..2,rot_0..3）逐值比对；
- ascii 格式小文件逐值比对；
- 尾随 face element 的 binary 文件应正确跳过、顶点逐值正确。
