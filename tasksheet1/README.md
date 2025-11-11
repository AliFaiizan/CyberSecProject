P1 – Boiler

P1_FT01 — 0–2,500 mmH₂O — measured flowrate of the return water tank. 

P1_FT02 — 0–2,500 mmH₂O — measured flowrate of the heating water tank. 

P1_FT03 — 0–2,500 mmH₂O — measured outflow rate of the return water tank. 

P1_LIT01 — 0–720 mm — water level of the return water tank. 

P1_PIT01 — 0–10 bar — heat‑exchanger outlet pressure. 

P1_PIT02 — 0–10 bar — water‑supply pressure of the heating‑water pump. 

P1_TIT01 — −50–150 °C — heat‑exchanger outlet temperature. 

P1_TIT02 — −50–150 °C — temperature of the heating water tank. 


P2 – Turbine

P2_24Vdc — 0–30 V — DCS 24 V input voltage (analog measurement). 

P2_SIT01 — 0–3,200 RPM — current turbine RPM measured by speed probe. 

P2_VT01 — 11–12 rad/s — key‑phasor probe phase‑lag signal. 

     P2_VIBTR01 (HAI 21.03 name: P2_VYT02) — −10–10 µm — Y‑axis displacement near the 1st mass wheel. 

    P2_VIBTR02 (HAI 21.03 name: P2_VXT02) — −10–10 µm — X‑axis displacement near the 1st mass wheel. 

    P2_VIBTR03 (HAI 21.03 name: P2_VYT03) — −10–10 µm — Y‑axis displacement near the 2nd mass wheel. 

    P2_VIBTR04 (HAI 21.03 name: P2_VXT03) — −10–10 µm — X‑axis displacement near the 2nd mass wheel. 

P3_LIT01 — 0–90 % — water level of the upper water tank. (Shown as “LIT01 (LT01)” in the table; the checkmark includes HAI 21.03.)

First, we loaded and cleaned the training and test datasets, ensuring that all samples under attack were excluded to maintain data integrity.
Sensor readings were normalized to allow for fair comparison between datasets. 
We then computed the Kolmogorov–Smirnov (K–S) statistic for each sensor to quantify distribution differences between the train and test sets. 
To assess system state coverage, we identified all unique actuator states in both datasets and calculated the percentage of states present in both sets, as well as the proportion of states in one set that are covered by the other. 
This approach provides insight into the representativeness and overlap of normal operating conditions between the training and test data,