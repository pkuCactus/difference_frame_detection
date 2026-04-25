# Ref 帧差异检测模块

## 1. 目标
检测当前输入图像帧与指定 Ref 参考图像帧的差异，确定是否送入后续事件分析，对于相似的不进入后续模块，降低无效计算。

## 2. 输入
- current_frame: cv::Mat
- ref_frame: cv::Mat
- similariity_threshold: float

## 3. 输出
- is_similar: bool

## 4. 规则
- 若ref_frame为空，则返回false，current_frame必须进入事件分析模块
- 若current_frame与ref_frame相似度>=similarity_threshold,则认为相同，返回true，否则认为不同返回false,默认阈值设置为0.85
- 相同则跳过，进入下一帧处理流程
- 不相同，则直接进入事件分析模块或者保存指定时长的视频然后进入事件分析模块
- 相似度支持配置，主要支持SSIM和像素差分两种，根据配置选择具体的

## 5. Ref更新策率
- 默认仅当current_frame成功进入事件分析模块，才更新ref_frame，跳过帧则不更新
- newest策略则是检测到有目标，则更新ref_frame否则不更新
- Ref 帧的更新策略需要支持这两种，然后根据配置进行选择具体的更新策略，默认采用newest的策略

## 6. 验收标准
- 首帧一定进入事件分析模块
- 相似帧被过滤跳过
- 差异帧进入事件分析模块
