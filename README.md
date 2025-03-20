# PURPLE: Dynamic Control for CAKE to Mitigate Bufferbloat

PURPLE is an algorithm designed to estimate "useful capacity" and control bufferbloat in wireless networks through dynamic adjustment of CAKE's bandwidth parameter. The algorithm draws inspiration from the BLUE active queue management scheme, itself being an evolution of RED, adapting its congestion detection mechanisms for wireless capacity estimation.

## Background

Bufferbloat, or excessive queuing delay under load, is a noticeable quality of service degradation that occurs when latency-sensitive traffic experiences the effects of increased packet buffering delays in network devices. This phenomenon leads to increased latency and reduced network performance, particularly affecting real-time applications.

While FQ-CoDel and CAKE are existing AQM solutions that help mitigate bufferbloat, they require properly configured bandwidth settings. In wireless networks where capacity can be highly variable, static bandwidth settings are insufficient. PURPLE addresses this by dynamically adjusting CAKE's bandwidth parameter based on real-time network conditions.

## How PURPLE Works

PURPLE consists of two main components:

1. Queue state monitoring
2. Capacity adjustment based on congestion signals
