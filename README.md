# ComfyUI 标定与尺寸估计节点（VP + Bird-eye）

该实现将你给出的流程拆成多个可连接节点，便于调试：

1. 图像预处理（去畸变/灰度化）
2. 直线检测（LSD/HoughLinesP）
3. 直线聚类（3方向聚类）
4. 消失点检测（每簇求交）
5. 焦距估计（三消失点正交约束）
6. 相机姿态恢复（R）
7. 地面单应矩阵（H）
8. 鸟瞰图生成（warpPerspective）
9. 尺寸计算（pixel->meter）

> 家具检测/分割节点按要求暂时跳过。

## 安装
将本目录放入 `ComfyUI/custom_nodes/` 下并重启 ComfyUI。

## 调试与报错信息
每个关键节点都提供 debug 字符串输出（json），并在失败时抛出带参数上下文的错误，便于定位：
- 参数不合法
- 直线太少
- 聚类失衡
- 消失点无穷远
- 焦距估计无正值候选
- 单应矩阵奇异

## 建议连接方式
`IMAGE -> ImagePreprocess -> LineDetection -> LineClustering -> VanishingPointDetection -> FocalLengthEstimation -> CameraPoseRecovery -> GroundHomography -> BirdEyeView`

尺寸计算可以单独接 `SizeMeasurementNode`，输入鸟瞰图上两点像素坐标与标定比例 `meters_per_pixel`。

## 直接打开的工作流文件
仓库已提供 `workflow_vp_birdeye.json`，可在 ComfyUI 里通过 **Load** 直接打开。

加载后请先在 `LoadImage` 节点选择你的图片，然后按需要调整：
- `LineDetectionNode` 的检测方法与阈值
- `LineClusteringNode` 的聚类角度
- `GroundHomographyNode` 的 `plane_height`
- `SizeMeasurementNode` 的两点坐标与 `meters_per_pixel`
