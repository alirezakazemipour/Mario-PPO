[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)  
# Mario-PPO
## 29 / 32 Levels Completed!

<p align="left">
  <img src="Demo/world1-stage1.gif" width="200">
  <img src="Demo/world1-stage2.gif" width="200">
  <img src="Demo/world1-stage3.gif" width="200">
  <img src="Demo/world1-stage4.gif" width="200"><br/>
  <img src="Demo/world2-stage1.gif" width="200">
  <img src="Demo/world2-stage2.gif" width="200">
  <img src="Demo/world2-stage3.gif" width="200">
  <img src="Demo/world2-stage4.gif" width="200"><br/>
  <img src="Demo/world3-stage1.gif" width="200">
  <img src="Demo/world3-stage2.gif" width="200">
  <img src="Demo/world3-stage3.gif" width="200">
  <img src="Demo/world3-stage4.gif" width="200"><br/>
  <img src="Demo/world4-stage1.gif" width="200">
  <img src="Demo/world4-stage2.gif" width="200">
  <img src="Demo/world4-stage3.gif" width="200"><br/>
  <img src="Demo/world5-stage1.gif" width="200">
  <img src="Demo/world5-stage2.gif" width="200">
  <img src="Demo/world5-stage3.gif" width="200">
  <img src="Demo/world5-stage4.gif" width="200"><br/>
  <img src="Demo/world6-stage1.gif" width="200">
  <img src="Demo/world6-stage2.gif" width="200">
  <img src="Demo/world6-stage3.gif" width="200">
  <img src="Demo/world6-stage4.gif" width="200"><br/>
  <img src="Demo/world7-stage1.gif" width="200">
  <img src="Demo/world7-stage2.gif" width="200">
  <img src="Demo/world7-stage3.gif" width="200"><br/>
  <img src="Demo/world8-stage1.gif" width="200">
  <img src="Demo/world8-stage2.gif" width="200">
  <img src="Demo/world8-stage3.gif" width="200"><br/>
</p>

## Documented Hyper-Parameters
World-Stage| rollout length| number of epochs| batch size| lr| gamma| lambda| entropy coefficient| clip range| number of workers| gradient clipping norm 
:---------:|:-------------:|:---------------:|:---------:|:---:|:---:|:----:|:------------------:|:---------:|:----------------:|:---------------------:
	1-1      |      128      |        8        |      64   |2.5e-4| 0.9| 0.95 |         0.01       |      0.2  |        8         |       0.5
  
## Acnowledgement
1. [@OpenAI](https://github.com/openai) for [Mario Wrapper](https://github.com/openai/large-scale-curiosity/blob/e0a698676d19307a095cd4ac1991c4e4e70e56fb/wrappers.py#L241).
2. [@uvipen](https://github.com/uvipen) for [Super-mario-bros-PPO-pytorch](https://github.com/uvipen/Super-mario-bros-PPO-pytorch).
3. [@roclark](https://github.com/roclark) for [Mario Reward](https://github.com/roclark/super-mario-bros-dqn/blob/2305549fe4a2eb273d98c3811b809bd9360e024a/core/wrappers.py#L110).
