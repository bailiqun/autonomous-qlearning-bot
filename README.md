#  autonomous-qlearning-bot
Autonomous car race game writen by pygame and qlearning.
###### pygame写小游戏，Agent自行学习如何获得最高的得分，无人驾驶的一个前戏工程。

<video src="https://github.com/bailiqun/autonomous-qlearning-bot/blob/master/example.mov" play></video>  

### 1 介绍
让汽车学习怎么避开障碍物属于强化学习（reforcement learning）范畴，可以用马尔科夫链来描述。强化学习中有状态(state)、动作(action)、奖赏(reward)这三个要素。
    
其中游戏的执行者（人、agent(game robot)、外星智慧体...)。抽象一下玩游戏的过程，执行者通过改变游戏状态获取奖罚的过程。
    这其中就包含
    
1. **状态(state)**:游戏人物中的位置坐标、敌人位置等 
2. **动作(action)**:游戏人物执行动作会改变人物状态
3. **奖罚(reward)**:游戏开始便一直伴随着状态的更替，比如得分、得装备、被歼灭、被玩弄


### 2 游戏

游戏场景就是一个小车躲避前方障碍物，有三条跑道每次会随机掉落两个障碍物，小车要从空车道通过便得一分。由此我们想让游戏自娱自乐，让他自己玩儿。于是我们使用QLearning方法，机器人自己学习自己的状态空间，最后你会发现他会比人玩儿的还溜。

###### 游戏界面

![image](https://github.com/bailiqun/autonomous-qlearning-bot/blob/master/intro.png)

#### 2.1 状态

本游戏状态如下图所示，在程序中以字符串"dx,dy",例如"200,100".则Q矩阵可以表示为字典，Q={"dx,dy":[0, 0, 0]}，状态对应的**动作**分别是['stay','left', 'right']表示机器人动作分别是不动、向左、向右。

![image](https://github.com/bailiqun/autonomous-qlearning-bot/blob/master/qstate.png)

#### 2.2 动作
参照 2.1

#### 2.3 奖罚
- 1 如果执行动作后存活（live）则reward = +1
- 2 如果执行动作后与障碍物碰撞(dead)，则reward = -100
- 3 如果车不在道路中间 (lane_center), 则reward = +10 (TODO)


