# benew_model

这里重新建立benew_model项目，为了更合理的管理模型相关的代码

- common 放置通用的脚本和代码
- sensor 放置sensor类
- flow 放置由sensors组装好的flow，都以flow_开头
- configure 放置配置文件、因子组文件等


一个基本模型，按顺序分为以下6个flow, 都是站在交易日的早晨, 计算最新的日子(大部分是昨日, 小部分是今日)
1. flow_pool, 有效的池子, 只和pool_name和benchmark_weight有关
2. flow_forward_return, 未来收益(存在未来数据), 和flow_pool, risk_factor_list有关
3. flow_ortho, alpha因子中性化且正交, 和flow_pool, risk_factor_list, alpha_factor_list有关
4. flow_estimation, 估计因子收益
5. flow_prediction_ortho, 预测股票未来收益等
6. flow_optim, 生成投资组合

#### optimization flow:
这个flow是从模型到投资组合，有如下不同的版本：
* flow_optim_v0.py:  传统的优化版本
* flow_optim_v0_forward_return.py: 基于v0，使用了未来数据做研究使用。
* flow_optim_v1.py: 动态transaction_cost的优化版本
* flow_optim_v1_forward_return: 基于flow_optim，使用了未来数据做研究使用。


#### 研究项目之FakeForwardReturn
* flow_fake_forward_return: 在生成forward_return之后，这个产生混入噪声的forward_return.
