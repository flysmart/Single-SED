# Single-SED
Practical for AI
<br><br><br><br>
**训练模型**<br>
数据预处理：运行`python preprocessing/preprocessingESC.py --csv_file ./data/ESC-50/meta/esc50.csv --data_dir  ./data/ESC-50/audio/ --store_dir ./predata/ --sampling_rate 44100`<br>
开始训练模型：运行`python train.py --config_path ./config/esc_densenet.json`<br>
<br><br><br><br>
**训练结果**<br><br>
![image](https://github.com/flysmart/Single-SED/assets/66983043/908bf9e5-5345-453b-b63b-7603b0ba524f#pic_center)<br>
![image](https://github.com/flysmart/Single-SED/assets/66983043/30afd1c9-f788-43c9-9e44-dec57774c7d6#pic_center)<br>
![image](https://github.com/flysmart/Single-SED/assets/66983043/41fda1e3-a0f0-4cba-adef-20a6bdcb20eb#pic_center)<br>
<br><br><br><br>
PS:源代码作者:[Rethinking CNN Models for Audio Classification](https://github.com/kamalesh0406/Audio-Classification)<br>
侵删
