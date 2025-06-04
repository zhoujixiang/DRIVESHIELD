# DRIVESHIELD: Scene Graph-Guided Multimodal Reasoning for Safer Autonomous Driving
_______________________________________________________________________________________
### Description: DRIVESHIELD is designed to monitor autonomous driving systems by leveraging multimodal large language models to proactively predict collision risks, understand complex vehicle interactions through scene graph representations, and generate clear, interpretable explanations along with actionable recovery strategies to enhance driving safety and system reliability.
_______________________________________________________________________________________
### General Python Setup
The ADS systems we use for testing are VAD and UniAD, both from Bench2Drive. The installation process follows the guidelines provided by [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive).

Once this setup is completed, install the rest of the requirements from requirements.txt:

```
pip install -r requirements.txt
```

### Usage Examples
#### Generate Scenario
You first need to generate the test scenarios.
If you want to run AVFUZZER in VAD, you should run:
```
bash /DRIVESHIELD/Bench2Drive/leaderboard/scripts/run_evaluation_multi_vad.sh
```
if you want run COLLVER in VAD, you should change config file run_evaluation.sh
#### Use DRIVESHIELD to Predict
Then you can run DRIVESHIELD, by 
```
python /DRIVESHIELD/VLM_Monitor/monitor.py
```
