# 相机标定照片放这里

1. 生成可打印棋盘格（A4、9x6 内角点、18mm 方格，板面 180x126mm，能放进 A4 且留足打印边距）：

   ```powershell
   python scripts/make_chessboard.py
   ```

   会生成 `assets/chessboard_9x6_18mm_A4.pdf` 和同名 PNG。

2. 打印时选择 **100% / 实际大小**（不要"适应页面"），打印后用尺子量一个方格，
   如果不是 18mm，把实际毫米数记下来（标定时 `--square` 传实测值）。
   把棋盘格贴在一块硬纸板/亚克力板上保持平整。

3. 用手机拍 12–15 张不同角度、不同距离的照片：正对、倾斜 30°/45°/60°、
   四角都拍；棋盘格完整出现在画面里，避免反光和运动模糊。

4. 照片拷到本目录（jpg/png 均可），然后运行：

   ```powershell
   python scripts/camera_calibration.py --square 18 --pattern 9x6
   ```

   输出 `reports/camera_calibration.json`（内参矩阵、畸变系数、重投影误差）
   和 `screenshots/calibration_undistort.jpg`（去畸变前后对比）。
   把 `rms_reprojection_error_px` 填进 README 和简历。

> 说明：`python scripts/camera_calibration.py --synthetic` 只用合成图像验证代码路径，
> 不能当作真实相机标定结果写进简历/README。
