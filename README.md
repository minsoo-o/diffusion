## Diffision Model 개념
### 전방 확산
```
x(t+dt) = x(t) + sigma(t)*sqrt(dt)*r
sigma = noise_strength_fn
r = random_variable N(0, 1)
```

### 역확산
```
x(t+dt) = x(t) + (sigma(T-t)**2)*score_fn*dt + sigma(T-t)*sqrt(dt)*r
score_fn = d/dxlog_p(x, t)
-> simple -> s(x, t) = -(x-x_0)*(sigma**2)*t = -x*(sigma**2)*t
```

score_fn을 알고 있으면, 역확산이 가능하다!
그런데, 실제로는 score_fn을 알지 못한다.
그래서 우리는 denoising network를 학습하는 과정에 score_fn을 배우게 할 것이다.
목표는, 확산 과정에서 모든 시간 t에 대해서, 그리고 모든 x_0에 대해서 샘플의 각 부분에 추가되는 노이즈의 양을 예측하는 것
score_fn을 학습하는 것은 무작위 노이즈를 의미있는 것으로 변환하는 것이다.


