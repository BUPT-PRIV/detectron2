## Debug
Train model base on local environment

ResNet-50

|                         |        V0.4         |        detectron2    |        log                                      |
| :---------------------: | :-----------------: | :------------------: | :---------------------------------------------: |  
| RPN-C4-1x-ms(impr)      |     51.5            | 51.5                 |<a href="docs/logs/rpn_c4_out_log.txt">output</a>|                       
| RPN-FPN-1x-ms(impr)     |     57.4            | 58.0                 |<a href="docs/logs/rpn_fpn_out_log.txt">output</a>| 
| Faster-C4-1x-ms(impr)   |     35.1            | 35.7                 |<a href="docs/logs/faster_c4_out_log.txt">output</a>|                     
| Faster-DC5-1x-ms(impr)  |     34.9            | || 

#### Experiments

| RPN-FPN-1x-ms(impr) | AR1000              | log                 |
| :-----------------: | :-----------------: | :-----------------: |
| + backbone          | 58.0                |<a href="docs/logs/rpn_fpn_out_backbone_log.txt">output</a>|
| + FPN               | 58.0                |<a href="docs/logs/rpn_fpn_out_backbone_fpn_log.txt">output</a>                     |
| + RPN               |                     |                     |
| + dataloder         |                     |                     |