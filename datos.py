import numpy as np
import pandas as pd
dias_por_mes = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
dias_por_mes_acum = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
dic_meses = {1:'Enero',
             2: 'Febrero',
             3: 'Marzo',
             4:'Abril',
             5: 'Mayo',
             6: 'Junio',
             7: 'Julio',
             8: 'Agosto',
             9: 'Septiembre',
             10: 'Octubre',
             11: 'Noviembre',
             12: 'Diciembre'}

datos = pd.read_excel('Datos_climatologicos_Santa_Fe_2019.xlsx', index_col=0)
