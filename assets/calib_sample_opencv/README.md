# 标定验证用样例图像（OpenCV 官方）

这 13 张是 **OpenCV 官方仓库** `samples/data/left01.jpg … left14.jpg` 的公开标定样例图像
（真实相机拍摄的棋盘格照片，640×480，9×6 内角点）。

- 来源：https://github.com/opencv/opencv/tree/4.x/samples/data
- 许可：OpenCV 仓库（Apache-2.0），此处仅用于在**真实照片**上验证标定流程，未作修改。
- 棋盘格物理边长未知，标定时按任意单位 `--square 1`；对平面靶标而言相机内参、畸变系数与
  重投影误差都与假设的方格尺寸无关（只有平移向量随尺度缩放）。

复现：

```powershell
python scripts/camera_calibration.py --images assets/calib_sample_opencv --pattern 9x6 --square 1 `
  --out reports/camera_calibration_sample.json
```

结果见 `reports/camera_calibration_sample.json`：13/13 视角成功提角，
RMS 重投影误差 0.231 px（单视角均值 0.182 px）。

> 注意：这是**别人的相机**的标定结果，只用来验证代码在真实照片上可用；
> 要标定自己的相机，把手机拍的照片放进 `../calib_photos/` 并运行
> `python scripts/camera_calibration.py --square <实测方格边长> --pattern 9x6`。
