haien_mapping_dict = {
    # ─────────────────────────────────────────────
    # Section 1001.xx-OUT  →  Boiler sensors (P1_Bxxxx)
    # ─────────────────────────────────────────────
    "1001.15-OUT": "P1_B2016",     # confirmed in table (page 15)

    # ─────────────────────────────────────────────
    # Section DM-FCVxx  →  Flow control valves
    # ─────────────────────────────────────────────
    "DM-FCV01-D": "P1_FCV01D",
    "DM-FCV01-Z": "P1_FCV01Z",
    "DM-FCV02-D": "P1_FCV02D",
    "DM-FCV02-Z": "P1_FCV02Z",
    "DM-FCV03-D": "P1_FCV03D",
    "DM-FCV03-Z": "P1_FCV03Z",

    # ─────────────────────────────────────────────
    # Section DM-FTxx  →  Flow transmitters
    # ─────────────────────────────────────────────
    "DM-FT01": "P1_FT01",
    "DM-FT01Z": "P1_FT01Z",
    "DM-FT02": "P1_FT02",
    "DM-FT02Z": "P1_FT02Z",
    "DM-FT03": "P1_FT03",
    "DM-FT03Z": "P1_FT03Z",

    # ─────────────────────────────────────────────
    # Section DM-LCV01  →  Level control valve
    # ─────────────────────────────────────────────
    "DM-LCV01-D": "P1_LCV01D",
    "DM-LCV01-Z": "P1_LCV01Z",
    # "DM-LCV01-MIS": "DQ03_LCV01-MIS",

    # ─────────────────────────────────────────────
    # Section DM-LTxx  →  Level transmitter
    # ─────────────────────────────────────────────
    "DM-LIT01": "P1_LIT01",

    # ─────────────────────────────────────────────
    # Section DM-PCVxx  →  Pressure control valves
    # ─────────────────────────────────────────────
    "DM-PCV01-D": "P1_PCV01D",
    "DM-PCV01-Z": "P1_PCV01Z",
    "DM-PCV02-D": "P1_PCV02D",
    "DM-PCV02-Z": "P1_PCV02Z",

    # ─────────────────────────────────────────────
    # Section DM-PTxx  →  Pressure transmitters
    # ─────────────────────────────────────────────
    "DM-PIT01": "P1_PIT01",
    "DM-PIT02": "P1_PIT02",
    "DM-PIT01-HH": "P1_PIT01_HH",      # pressure of heating-water tank TK02

    # ─────────────────────────────────────────────
    # Section DM-PPxx  →  Pumps
    # ─────────────────────────────────────────────
    "DM-PP01A-D": "P1_PP01AD",     # start command main water pump A
    "DM-PP01A-R": "P1_PP01AR",     # running state  main water pump A
    "DM-PP01B-D": "P1_PP01BD",
    "DM-PP01B-R": "P1_PP01BR",
    "DM-PP02-D": "P1_PP02D",
    "DM-PP02-R": "P1_PP02R",
    # "DM-PP04-AO": "P1_PP04AO",     # cooling-pump speed
    # "DM-PP04-D": "P1_PP04D",

    # ─────────────────────────────────────────────
    # Section DM-SOLxx  →  Solenoid valves
    # ─────────────────────────────────────────────
    "DM-SOL01-D": "P1_SOL01D",     # main-tank (TK01) supply valve
    # "DM-SOL02-D": "P1_SOL02D",     # return-tank (TK03) supply valve
    "DM-SOL03-D": "P1_SOL03D",     # main-tank drain valve
    # "DM-SOL04-D": "P1_SOL04D",     # return-tank drain valve

    # ─────────────────────────────────────────────
    # Section DM-ST / SW  →  Boiler control
    # ─────────────────────────────────────────────
    "DM-ST-SP": "P1_STSP",         # start/stop boiler DCS
    # "DM-SW01-ST": "P1_SW01ST",     # start command (control panel)
    # "DM-SW02-SP": "P1_SW02SP",     # stop command (control panel)
    # "DM-SW03-EM": "P1_SW03EM",     # emergency stop (control panel)

    # ─────────────────────────────────────────────
    # Section DM-TTxx  →  Temperature transmitters
    # ─────────────────────────────────────────────
    "DM-TIT01": "P1_TIT01",         # heat-exchanger outlet temp
    # "DM-TIT02": "P1_TIT02",         # heating-water tank temp
    "DM-TIT03": "P1_TIT03",         # main-water tank temp
    # "DM-TT04": "P1_TIT04",         # return-water tank temp
    # "DM-TT05": "P1_TIT05",         # buffer-water tank temp

    # ─────────────────────────────────────────────
    # Section GATEOPEN / PP04-SP-OUT  →  Steam system
    # ─────────────────────────────────────────────
    "GATEOPEN": "P4_ST_GOV",       # gate opening of STM
    # "PP04-SP-OUT": "P4_ST_TEMP"    # running temperature of cooling
}
