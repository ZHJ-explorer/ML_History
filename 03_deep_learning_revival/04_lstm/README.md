# LSTM（长短期记忆网络）

## 历史背景

1997年，Hochreiter和Schmidhuber提出了LSTM（Long Short-Term Memory）网络，解决了传统RNN的梯度消失问题。LSTM通过引入门控机制，能够学习长期依赖关系。

**当时的问题**：
- 传统RNN难以学习长序列依赖
- 梯度消失导致早期输入信息丢失
- 训练深层RNN困难

**核心突破**：
- 遗忘门、输入门、输出门三个门控机制
- 细胞状态作为信息高速公路

## 数学原理

### LSTM单元结构

$$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$$
$$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$
$$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$$
$$\tilde{C}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
$$h_t = o_t \odot \tanh(C_t)$$

### 门控机制

- **遗忘门**：决定丢弃多少历史信息
- **输入门**：决定更新多少新信息
- **输出门**：决定输出多少当前信息

## 历史局限性

1. 计算复杂度高
2. 参数较多
3. 后被GRU等简化版本部分取代

## 历史影响

- 开启了序列建模的新时代
- 应用于机器翻译、语音识别等
- 启发了Transformer的发展

## 思考题

1. LSTM为什么能解决梯度消失问题？
2. GRU和LSTM有什么区别？
3. LSTM在现代NLP中的角色是什么？