# 相机人体检测能力探测模块

## 1. 目标

系统启动时，根据配置的相机信息，探测该相机是否支持通过接口返回人体目标检测结果。

探测结果将决定系统进入哪条处理链路：
- **支持**：进入相机检测结果获取模式，周期性查询相机检测结果并拉流解码
- **不支持**：进入本地检测模式，直接拉流解码并在 RK3566 上执行本地人体检测

---

## 2. 输入

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `camera_id` | `std::string` | 是 | 相机唯一标识 |
| `camera_host` | `std::string` | 是 | 相机 IP 地址或域名 |
| `camera_port` | `int` | 是 | 相机服务端口 |
| `capability_url` | `std::string` | 是 | 能力探测接口路径 |
| `timeout_ms` | `int` | 是 | 接口调用超时时间，单位毫秒 |
| `protocol` | `std::string` | 是 | 接口协议类型，取值 `ONVIF` 或 `REST` |

> 以上参数均从 YAML 配置文件中读取。

---

## 3. 输出

| 字段 | 类型 | 说明 |
|------|------|------|
| `supported` | `bool` | 相机是否支持返回人体检测结果 |
| `reason` | `std::string` | 判断依据说明，用于日志记录 |

---

## 4. 判断规则

| 条件 | 判定结果 | 说明 |
|------|----------|------|
| 接口调用成功，且返回 `supported = true` | 支持 | 进入相机检测结果获取模式 |
| 接口调用成功，但返回 `supported = false` | 不支持 | 相机明确声明不支持 |
| 接口调用超时 | 不支持 | 超时视为能力不可用 |
| 接口返回非预期格式 | 不支持 | 解析失败视为不支持 |
| 网络连接失败 | 不支持 | 无法到达相机 |

- 当判定为不支持时，系统自动切换到本地检测模式
- 能力探测仅在系统启动或重连时执行一次，运行过程中不重复探测

---

## 5. 异常处理

| 异常场景 | 处理方式 | 日志级别 |
|----------|----------|----------|
| 接口调用超时 | 返回不支持，进入本地检测模式 | `WARNING` |
| HTTP 响应状态码非 200 | 返回不支持，记录状态码 | `WARNING` |
| 返回数据 JSON/XML 解析失败 | 返回不支持，记录原始响应内容 | `ERROR` |
| 网络不可达 | 返回不支持，记录连接错误信息 | `ERROR` |

- 所有异常场景均不得阻塞主流程启动
- 异常时应在日志中记录 `camera_id`、异常类型、错误信息

---

## 6. 接口适配说明

系统需支持两种接口协议获取能力探测结果，通过配置项 `protocol` 切换：

### REST 模式
- 请求方式：`GET`
- 请求地址：`http://{camera_host}:{camera_port}/{capability_url}`
- 返回格式：JSON
- 返回示例：
  ```json  
  {  
    "supported": true  
  }  

### ONVIF 模式
- 通过 ONVIF 标准接口查询设备能力
- 解析返回的能力描述中是否包含人体检测分析能力
- 具体接口调用方式需参考 ONVIF Analytics Service 规范

### 当前版本默认不进行鉴权

### 接口定义（当前版本）

当前版本采用预定义的空接口实现：

```cpp
class ICameraCapabilityChecker {
public:
    virtual ~ICameraCapabilityChecker() = default;
    
    // 当前版本直接返回配置的支持状态，不实际调用相机接口
    // 后续联调时再对接真实的 ONVIF/REST 接口
    virtual bool isSupportDetection() = 0;
};
```

- 默认实现：根据配置文件中的 `camera_detection.enabled` 字段直接返回支持或不支持
- 后续联调时再实现真实的 ONVIF/REST 接口调用

## 7. 验收标准

#### 场景 1：相机支持人体检测
- Given 相机能力探测接口可正常访问，且返回 supported = true
- When 系统启动并执行能力探测
- Then 系统进入相机检测结果获取模式
- And 日志中记录：[INFO] camera_id=xxx,detection_supported=true, mode=camera_detection
#### 场景 2：相机不支持人体检测
- Given 相机能力探测接口返回 supported = false
- When 系统启动并执行能力探测
- Then 系统进入本地检测模式
- And 日志中记录：[INFO] camera_id=xxx, detection_supported=false, mode=local_detection
#### 场景 3：接口调用超时
- Given 相机能力探测接口在配置的超时时间内未响应
- When 系统启动并执行能力探测
- Then 系统进入本地检测模式
- And 日志中记录：[WARNING] camera_id=xxx, capability check timeout, fallback to local_detection
#### 场景 4：接口返回异常数据
- Given 相机接口返回的数据无法正常解析
- When 系统启动并执行能力探测
- Then 系统进入本地检测模式
- And 日志中记录：[ERROR] camera_id=xxx, capability response parse failed, fallback to local_detection

## 9. 补充说明
- 本模块仅负责能力探测，不负责后续检测结果的持续获取
- 能力探测结果将传递给主流程状态机，由状态机决定进入哪条处理链路
- 当 RTSP 断流重连成功后，建议重新执行一次能力探测
