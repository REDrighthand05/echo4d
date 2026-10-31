# DS Flash 小函数规格模板（量产零件用）

每张工单只给 DS Flash 写**一个**强限定函数。规格必须钉死以下六项，
缺一项不发。产出由 Muse 逐行审 + 跑自测后才注册进 `echo_super`。

```
函数名:        <snake_case，全局唯一>
consumes:      [<EchoObject.kind>, ...]        # 只吃这些，多一个不要
produces:      <EchoObject.kind>               # 只吐这一个
签名:          def <name>(<obj1>, <obj2>) -> <payload 类型>
               # 入参是 EchoObject，函数内只读 .data / .t / .frame
               # 返回裸 payload 即可，包装由集成层做
性能要求:      <如 1080p 单帧 < 2s CPU / 全向量化禁逐像素 Python 循环>
禁止事项:      不读全局状态、不写文件（除非签名里给了路径）、
               不 import 未在清单里的库、不吞异常
失败方式:      输入不合规格时 raise ValueError 并说明哪项不对
自测素材:      <给出 1 个输入样例和期望输出的关键性质，不给实现提示>
```

## 为什么这样能近零成本

- 规格窄 → DS 不漂移，写错整张扔掉重出，不修修补补。
- 只声明 consumes/produces → 集成层自动接管，函数之间互不知情。
- 统一对象包装 → 新函数挂上即被超级函数发现可用，接口永不变。
