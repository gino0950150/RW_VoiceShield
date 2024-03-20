Deep learning techniques have shown promising results in automatic respiratory sound classification. However, distinguishing respiratory sounds in real world noisy conditions pose challenges for the system to be used in clinical settings. Instead of noise injection augmentation that is conventionally done, we propose an audio enhancement (AE) pipeline prior to the respiratory sound classification system. Our AE pipeline is an adversarial network that is trained on real-world clinical noise.
Integrating this pipeline improved the ICBHI classification score by 4.24% on ICBHI respiratory sound dataset and by 3.57% on our recently-collected Formosa Archive of Breath Sounds (FABS) in multi-class noisy scenarios, compared to the baseline method of noise injection data augmentation. More importantly, the enhanced audio aids adoption by clinicians. In our physician validation study, we quantitatively demonstrate improvements in efficiency, diagnostic confidence, and trust during model-assisted diagnosis with our system over raw noisy recordings. Workflows integrating enhanced audio increased 11.61% diagnostic sensitivity and reached high-confidence diagnoses. Our system showcases audio enhancement as an effective methodology for increasing robustness and clinical utility of AI-assisted respiratory sound analysis. 

## Normal: 

We recommend using headphones for this section.

|          | Target                                                                    |Noisy                                                                      |  MANNER  | CMGAN|
|----------|---------------------------------------------------------------------------|---------------------------------------------------------------------------|----------|------|
| 0| ![](samples/Normal/N0_clean.png)                                         | ![](samples/Normal/N0_noisy.png)                                         | ![](samples/Normal/N0_MANNER.png)                                         |![](samples/Normal/N0_CMGAN.png)                                         |
|    | <audio src="samples/Normal/N0_clean.wav" controls="" preload=""></audio> | <audio src="samples/Normal/N0_noisy.wav" controls="" preload=""></audio> |<audio src="samples/Normal/N0_MANNER.wav" controls="" preload=""></audio>|<audio src="samples/Normal/N0_CMGAN.wav" controls="" preload=""></audio>|
| 1 | ![](samples/Normal/N1_clean.png)                                         | ![](samples/Normal/N1_noisy.png)                                         | ![](samples/Normal/N1_MANNER.png)                                         |![](samples/Normal/N1_CMGAN.png)                                         |
|    | <audio src="samples/Normal/N1_clean.wav" controls="" preload=""></audio> | <audio src="samples/Normal/N1_noisy.wav" controls="" preload=""></audio> |<audio src="samples/Normal/N1_MANNER.wav" controls="" preload=""></audio>|<audio src="samples/Normal/N1_CMGAN.wav" controls="" preload=""></audio>|
| 2 | ![](samples/Normal/N2_clean.png)                                         | ![](samples/Normal/N2_noisy.png)                                         | ![](samples/Normal/N2_MANNER.png)                                         |![](samples/Normal/N2_CMGAN.png)                                         |
|    | <audio src="samples/Normal/N2_clean.wav" controls="" preload=""></audio> | <audio src="samples/Normal/N2_noisy.wav" controls="" preload=""></audio> |<audio src="samples/Normal/N2_MANNER.wav" controls="" preload=""></audio>|<audio src="samples/Normal/N2_CMGAN.wav" controls="" preload=""></audio>|


## Crackles:

We recommend using headphones for this section.

|          | Target                                                                    |Noisy                                                                      |  MANNER  | CMGAN|
|----------|---------------------------------------------------------------------------|---------------------------------------------------------------------------|----------|------|
| 0| ![](samples/Crackle/C0_clean.png)                                         | ![](samples/Crackle/C0_noisy.png)                                         | ![](samples/Crackle/C0_MANNER.png)                                         |![](samples/Crackle/C0_CMGAN.png)                                         |
|    | <audio src="samples/Crackle/C0_clean.wav" controls="" preload=""></audio> | <audio src="samples/Crackle/C0_noisy.wav" controls="" preload=""></audio> |<audio src="samples/Crackle/C0_MANNER.wav" controls="" preload=""></audio>|<audio src="samples/Crackle/C0_CMGAN.wav" controls="" preload=""></audio>|
| 1 | ![](samples/Crackle/C1_clean.png)                                         | ![](samples/Crackle/C1_noisy.png)                                         | ![](samples/Crackle/C1_MANNER.png)                                         |![](samples/Crackle/C1_CMGAN.png)                                         |
|    | <audio src="samples/Crackle/C1_clean.wav" controls="" preload=""></audio> | <audio src="samples/Crackle/C1_noisy.wav" controls="" preload=""></audio> |<audio src="samples/Crackle/C1_MANNER.wav" controls="" preload=""></audio>|<audio src="samples/Crackle/C1_CMGAN.wav" controls="" preload=""></audio>|
| 2 | ![](samples/Crackle/C2_clean.png)                                         | ![](samples/Crackle/C2_noisy.png)                                         | ![](samples/Crackle/C2_MANNER.png)                                         |![](samples/Crackle/C2_CMGAN.png)                                         |
|    | <audio src="samples/Crackle/C2_clean.wav" controls="" preload=""></audio> | <audio src="samples/Crackle/C2_noisy.wav" controls="" preload=""></audio> |<audio src="samples/Crackle/C2_MANNER.wav" controls="" preload=""></audio>|<audio src="samples/Crackle/C2_CMGAN.wav" controls="" preload=""></audio>|


## Wheezes: 

We recommend using headphones for this section.

|          | Target                                                                    |Noisy                                                                      |  MANNER  | CMGAN|
|----------|---------------------------------------------------------------------------|---------------------------------------------------------------------------|----------|------|
| 0| ![](samples/Wheeze/W0_clean.png)                                         | ![](samples/Wheeze/W0_noisy.png)                                         | ![](samples/Wheeze/W0_MANNER.png)                                         |![](samples/Wheeze/W0_CMGAN.png)                                         |
|    | <audio src="samples/Wheeze/W0_clean.wav" controls="" preload=""></audio> | <audio src="samples/Wheeze/W0_noisy.wav" controls="" preload=""></audio> |<audio src="samples/Wheeze/W0_MANNER.wav" controls="" preload=""></audio>|<audio src="samples/Wheeze/W0_CMGAN.wav" controls="" preload=""></audio>|
| 1 | ![](samples/Wheeze/W1_clean.png)                                         | ![](samples/Wheeze/W1_noisy.png)                                         | ![](samples/Wheeze/W1_MANNER.png)                                         |![](samples/Wheeze/W1_CMGAN.png)                                         |
|    | <audio src="samples/Wheeze/W1_clean.wav" controls="" preload=""></audio> | <audio src="samples/Wheeze/W1_noisy.wav" controls="" preload=""></audio> |<audio src="samples/Wheeze/W1_MANNER.wav" controls="" preload=""></audio>|<audio src="samples/Wheeze/W1_CMGAN.wav" controls="" preload=""></audio>|
| 2 | ![](samples/Wheeze/W2_clean.png)                                         | ![](samples/Wheeze/W2_noisy.png)                                         | ![](samples/Wheeze/W2_MANNER.png)                                         |![](samples/Wheeze/W2_CMGAN.png)                                         |
|    | <audio src="samples/Wheeze/W2_clean.wav" controls="" preload=""></audio> | <audio src="samples/Wheeze/W2_noisy.wav" controls="" preload=""></audio> |<audio src="samples/Wheeze/W2_MANNER.wav" controls="" preload=""></audio>|<audio src="samples/Wheeze/W2_CMGAN.wav" controls="" preload=""></audio>|
