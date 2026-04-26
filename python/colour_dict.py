
# dictionary for assigning variable colours
# peak/all used to colour peak demand and total demand
# var1/var2 are a similar set (binary colouring etc)
# var3/4/5 are a similar set
demand_cols = {'peak': 'orange', 'all': 'tomato', 
                'var1': 'darksalmon', 'var2': 'salmon', 
                'var3': 'peru', 'var4': 'sienna', 'var5': 'brown', }

# dictionary for variable axis names when using all total demand data
var_dict_total = {'rainfall': 'Daily Rainfall (mm)', 'pv_capacity': 'PV Installation', 'temperature': 'Temperature (\u00b0C)', 'solar_power': r'Solar Irradiance (Wh m$^{-2}$)', 'total_demand': 'Total Power (MW)'}

# dictionary for variable axis names when using conditional peak demand data
var_dict_peak = {'rainfall': 'Daily Rainfall (mm)', 'pv_capacity': 'PV Installation', 'temperature': 'Temperature (\u00b0C)', 'solar_power': r'Solar Irradiance (Wh m$^{-2}$)', 'total_demand': 'Peak Daily Power (MW)'}
