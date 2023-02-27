# VLSI-DSP-Final-Project
Implementation if several MIMO detection algorithm via comparison
## System Model
![image](https://user-images.githubusercontent.com/114923630/221602194-c23a011b-2a40-4aa9-b2f6-0d81e350914c.png)
## MIMO detection algorithms
* ML detector (optimal)
* ZF detector
* MMSE detector
* $k$-best detector

![image](https://user-images.githubusercontent.com/114923630/221602916-accbf8d7-e343-4a9f-8fcb-75846e685896.png)
![image](https://user-images.githubusercontent.com/114923630/221602988-df319fcd-5372-43bd-b2fa-01583a661921.png)
## Simulation results
* Detection Results
![image](https://user-images.githubusercontent.com/114923630/221603920-f1050fdf-1006-466d-98f3-7b2ebf042d0e.png)

* Case study of the noise enhancement problem in ZF detector
![image](https://user-images.githubusercontent.com/114923630/221603943-72cd43fe-fdf2-4244-bec4-15bb0fed44f5.png)

* BER performance comparison
![image](https://user-images.githubusercontent.com/114923630/221604074-7d29bd5d-962a-41eb-81c9-8ae99ccd0d47.png)

## Conclusion
* From the simulation results, we can found the disadvantage of ZF detector
  * Noise enhancement problem may influence the detection results seriously
* For the linear detector
  * ZF is the best linear detector regarding SNR criterion
  * MMSE is the best linear detector regarding SINR criterion
* For the $k$-best detector
  * Has a higher performance than linear detector but also higher complexity
* The selection of $k$ in $k$-best detector is an important issue
  * Since the value of $k$ is a tradeoff between BER and complexity
* $k$-best detector is a more flexible choice than ZF and MMSE for MIMO system and VLSI implementation
* Summary of all detectors
  |Detector|BER|Complexity|Type|
  |---|---|---|---|
  |ML|Optimal|Dramically complex||
  |ZF|Very poor|Very simple|Linear detector|
  |MMSE|Poor|Simple|Linear detector|
  |$k$-best|Good|Rather complex|Tree-based-search detector (breadth first search)|



