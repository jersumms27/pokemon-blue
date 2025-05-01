# Deep Q-Learning Algorithm (huggingface.co)

Initialize replay memory $D$ to capacity $N$

Initialize action-value function $Q$ with random weights $\theta$

Initialize target action-value function $\hat{Q}$ with weights $\hat{\theta}=\theta$

**For** episode $- 1, M$ **do**

&nbsp; &nbsp; Initialize sequence $s_1=\{x_1\}$ and preprocessed sequence $\phi_1=\phi(s_1)$

&nbsp; &nbsp; **For** $t=1$,$T$ **do**

&nbsp; &nbsp; &nbsp; &nbsp; With probability $\epsilon$ select a random action $a_t$

&nbsp; &nbsp; &nbsp; &nbsp; otherwise select $a_t = argmax_{a}Q(\phi(s_t),a;\theta)$

&nbsp; &nbsp; &nbsp; &nbsp; Execute action $a_t$ in emulator and observe reward $r_t$ and image $x_{t+1}$

&nbsp; &nbsp; &nbsp; &nbsp; Set $s_{t+1}=s_t$,$a_t$,$x_{t+1}$ and preprocess $\phi_{t+1}=\phi(s_{t+1})$

&nbsp; &nbsp; &nbsp; &nbsp; Store transition ($\phi_t$,$a_t$,$r_t$,$\phi_{t+1}$) in $D$

&nbsp; &nbsp; &nbsp; &nbsp; Sample random minibatch of transitions ($\phi_j$,$a_j$,$r_j$,$\phi_{j+1}$) from $D$

&nbsp; &nbsp; &nbsp; &nbsp; Set $y_j=r_j$ if episode terminates at step $j+1$,
$r_j+\gamma\:max_{a'}\hat{Q}(\phi_{j+1},a';\bar{\theta})$ otherwise

&nbsp; &nbsp; &nbsp; &nbsp; Perform a gradient descent step on $(y_j-Q(\phi_j,a_j;\theta))^2$ with respect to the network parameters $\theta$

&nbsp; &nbsp; &nbsp; &nbsp; Every C steps reset $\hat{Q}=Q$

&nbsp; &nbsp; **End For**

**End For**

## Loss function

Q-Target: $y_j=r_j+\gamma\:max_{a'}\hat{Q}(\phi{j+1},a';\bar{\theta})$

Q-Loss: $y_j-Q(\phi_j,a_j;\theta)$