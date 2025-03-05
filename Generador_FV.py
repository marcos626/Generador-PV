# Actividad 2: Generador FV (módulo con definición de clase)
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from datos import datos, dic_meses, dias_por_mes_acum, dias_por_mes


class Generador_FV:
    """
    Instalación fotovoltaica con seguidor del punto de máxima potencia.
    """
    def __init__(self, tabla_anual, N, Ppico, eta, kp, Pinv, mu=2, Gstd=1000, Tr=25):
        """
        Método constructor.

        tabla anual: Contenedor con las mediciones de irradiancia y temperatura ambiente.
        N: Cantidad de paneles de la instalación.
        Ppico: Potencia pico de cada panel en Watt.
        eta: Rendimiento global de la instalación, por unidad.
        kp: Coef. de temperatura-potencia en p.u./Celsius.
        Pinv: Potencia nominal del inversor en kW.
        """
        self.tabla_anual = tabla_anual
        self.N = N
        self.Ppico = Ppico
        self.eta = eta
        self.kp = kp
        self.Pinv = Pinv
        self.mu = mu  # Umbral porcentual mínimo del inversor
        self.__Gstd = Gstd  # Irradiancia estándar, en W/m2. Atributo privado
        self.__Tr = Tr   # Temperatura de referencia, en Celsius. Atributo privado

    @property  # decorador getter para acceder al atributo privado __Gstd
    def Gstd(self):
        """Getter para el atributo privado __Gstd"""
        return self.__Gstd
    
    @property  # decorador getter para acceder al atributo privado __Tr
    def Tr(self):
        """Getter para el atributo privado __Tr."""
        return self.__Tr
        
    def __str__(self):
        """
        Mensaje a mostrar con print()
        """
        mensaje = 'Instalación fotovoltaica con una potencia ' + \
                  f'nominal de {self.N * self.Ppico * 1e-3} kW'
        return mensaje
    
    def __repr__(self):
        return f'<Generador FV: N={self.N}, Ppico={self.Ppico}, eta={self.eta}, ' + \
            f'kp={self.kp}, Pinv={self.Pinv}, mu={self.mu}>'

    @staticmethod
    def calcular_fila(d, m, h, mi):  # función helper
        """Convierte la fecha a un número de fila que va de 0 a 52559
        (1ero de enero a las 00:00 hs hasta el 31 de diciembre a las 23:50)"""
        dia_del_anio = dias_por_mes_acum[m - 1] + d
        fila = (dia_del_anio - 1) * 24 * 6 + (h * 6 + mi // 10)
        return fila

    @staticmethod
    def calcular_fecha_desde_fila(fila):  # función helper
        """Convierte un número de fila (0 a 52559) en una tupla día, mes, hora y minuto"""
        total_minutos = fila * 10
        dia_del_anio = total_minutos // (24 * 60) + 1  # Día del año (ajustando desfase)
        minutos_restantes = total_minutos % (24 * 60)
        hora = minutos_restantes // 60
        minuto = minutos_restantes % 60
        mes = 1
        for i in range(1, len(dias_por_mes_acum)):
            if dia_del_anio <= dias_por_mes_acum[i]:
                mes = i
                break
        else:
            mes = 12  # Si el día es después del acumulado final, es diciembre.

        # Determinar día del mes
        dia = dia_del_anio - dias_por_mes_acum[mes - 1]
        tupla = dia, mes, hora, minuto
        return tupla
    
    @staticmethod
    def momento(tupla):  # función usada en la notebook
        """Recibe una tupla = (d, m, h y mi) y devuelve un mensaje con la fecha correspondiente.
        d: Día del mes.
        m: Número del mes (de 1 a 12).
        h: Hora del día (de 0 a 23).
        mi: Minuto (puede valer 0, 10, 20, 30, 40 o 50)"""
        mensaje = (f'{tupla[0]}/{tupla[1]} a las {tupla[2]:02d}:{tupla[3]:02d} hs')
        return mensaje
        
    def irrad_temp(self, d, m, h, mi):
        """Devuelve un contenedor con 2 elementos extraídos de la tabla de datos medidos, 
        irradiancia y temperatura ambiente para el momento indicado, en este orden.
        # d: día; m: mes del año; h: hora del día; mi: minuto (0, 10, 20, 30, 40 o 50)"""
        return self.tabla_anual[self.calcular_fila(d, m, h, mi)]

    def irrad_temp_rango(self, tupla1, tupla2):
        """
        Similar a irrad_temp, pero devuelve una lista de tuplas con todas las filas entre los momentos indicados
        por tupla1 y tupla2. Estas 2 tuplas presentan el formato (d, m, h, mi).
        Por ej., si tupla1 = (24, 4, 10, 20) y tupla2 = (26, 6, 13, 0), el contenedor que se devuelve
        contiene la info de irradiancia y temp. para todas las mediciones desde el 24/04 a las 10:20 hs,
        hasta el 26/06 a las 13:00 hs. """
        d1, m1, h1, mi1 = tupla1
        d2, m2, h2, mi2 = tupla2
        inicio = self.calcular_fila(d1, m1, h1, mi1)
        fin = self.calcular_fila(d2, m2, h2, mi2)
        if inicio > fin:
            raise ValueError("La fecha inicial debe ser anterior a la fecha final")
        intervalo = [(G, T) for i, (G, T) in enumerate(self.tabla_anual[inicio:fin + 1])]  # f(x) para todo valor de x en X
        return intervalo

    def pot_modelo_GFV(self, G, T):
        """
        Devuelve la potencia generada por el GFV cuando la irradiancia es G y la temp. ambiente es T.
        G: Irradiancia global normal a la superficie de los módulos fotovoltáicos. [W/m2]
        T: Temperatura ambiente en celsius.
        """
        Tc = T + 0.031 * G  # temperatura de la celda
        P = self.N * (G / self.__Gstd) * self.Ppico * (1 + self.kp * (Tc - self.__Tr)) * self.eta * 1e-3
        #print(f'La potencia calculada es {P:.2f}')
        Pmin = (self.mu / 100) * self.Pinv
        P = P * (Pmin <= P < self.Pinv) + self.Pinv * (P > self.Pinv)
        return P

    def pot_generada(self, tupla):
        """
        Devuelve la potencia generada por un GFV. Recibe una tupla indicando el instante,
        en lugar de G y T como lo hace pot_modelo_GFV.
        tupla_instante: (d, m, h, mi)
        """
        d, m, h, mi = tupla
        G, T = self.irrad_temp(d, m, h, mi)
        P = self.pot_modelo_GFV(G, T)
        return P

    def pot_generada_rango(self, tupla1, tupla2):
        """
        Devuelve un contenedor con las potencias generadas en todos los instantes de medición 
        en el rango especificado por tupla1 y tupla2.
        Cada tupla se escribe en el formato (d, m, h, mi).
        """
        intervalo = self.irrad_temp_rango(tupla1, tupla2)
        p_rango = [self.pot_modelo_GFV(G, T) for G, T in intervalo]  # = f(x) para todo valor de x in X
        return p_rango

    def pot_media_mes(self, mes):
        """Devuelve la potencia media entregada en el mes especificado (en kW)"""
        tupla1 = (1, mes, 0, 0)
        tupla2 = (dias_por_mes[mes - 1], mes, 23, 50)
        p_rango = self.pot_generada_rango(tupla1, tupla2)
        # p_total = sum(p_rango)
        # p_media_mensual = (p_total) / (dias_por_mes[mes - 1] * 24 * 6)
        p_media_mensual = np.mean(p_rango)
        return p_media_mensual

    def pot_media_anual(self):
        """Devuelve la potencia media anual del GFV, en kW"""
        tupla1 = (1, 1, 0, 0)
        tupla2 = (31, 12, 23, 50)
        p_rango = self.pot_generada_rango(tupla1, tupla2)
        p_media_anual = np.mean(p_rango)
        return p_media_anual

    def energia_mes(self, mes):
        """Devuelve la energía entregada por el GFV en el mes indicado, medida en kWh (kilo-Watt-hora)"""
        p_media = self.pot_media_mes(mes)
        energia_mensual = p_media * dias_por_mes[mes - 1] * 24
        return energia_mensual

    def energia_anual(self):
        """Devuelve la energía entregada por el GFV en el año, en kWh"""
        p_media_anual = self.pot_media_anual()
        energia_anual = p_media_anual * 365 * 24
        return energia_anual

    def factor_de_utilizacion(self):
        """
        Devuelve el factor de utilización anual de la instalación.
        Se calcula como la proporción de la energía entregada en el año, en relación a la que podría
        haber entregado si todo el tiempo hubiera desarrollado la potencia nominal del inversor.
        """
        energia_generada = self.energia_anual()
        energia_ideal = 2.5 * 365 * 24
        f_utilizacion = energia_generada / energia_ideal
        return f_utilizacion

    def max_energ_mes(self):
        """
        Devuelve una tupla con 2 valores. El primero de ellos es el mes
        de máxima producción energética, y el segundo es la energía obtenida, en kWh.
        """
        max_energia = 0
        mes_max = 0
        for mes in dic_meses.keys():
            energia = self.energia_mes(mes)
            #print(f'Energia mensual {energia} de {dic_meses[mes]}')
            if energia > max_energia:
                max_energia = energia
                mes_max = mes
        return (mes_max, max_energia)

    def max_pot_mes(self):
        """
        Devuelve una tupla de 2 elementos. El primero es el mes para el cual se identifica
        la potencia máxima entregada, y el segundo es el valor de la potencia obtenida, en kW.
        """
        tupla1 = (1, 1, 0, 0)
        tupla2 = (31, 12, 23, 50)
        p_rango = self.pot_generada_rango(tupla1, tupla2)
        max_potencia = 0
        cont = 1
        for pot in p_rango:
            if pot > max_potencia:
                max_potencia = pot
                fecha = self.calcular_fecha_desde_fila(cont)
            cont += 1
        mes_max = fecha[1]  # (d, m, h, mi) extrae el día
        return (mes_max, max_potencia)

    def graficar_pot_rango(self, tupla1, tupla2):
        """Genera una gráfica interactiva con la variación temporal de la potencia generada."""
        d1, m1, h1, mi1 = tupla1
        d2, m2, h2, mi2 = tupla2
        pot = self.pot_generada_rango(tupla1, tupla2)
        t = np.linspace(0, len(pot), len(pot))  # Crear eje de tiempo

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=pot, mode='lines', line=dict(color='royalblue'), name="Potencia [kW]"))

        fig.update_layout(title=f'Potencia generada entre {d1}/{m1} {h1}:{mi1} hs y {d2}/{m2} {h2}:{mi2} hs',
                        xaxis_title="Tiempo", yaxis_title="Potencia [kW]", template="seaborn", hovermode="x") # opciones: 'plotly' o 'seaborn', 'plotly_dark'
        st.plotly_chart(fig, use_container_width=True)

    def graficar_energia_mensual(self):
        """Traza un diagrama de barras con la energía obtenida en cada mes del año usando Plotly"""
        energia = [round(self.energia_mes(mes), 2) for mes in dic_meses.keys()]
        t = list(dic_meses.values())  # Nombres de los meses
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=t, y=energia, marker_color='royalblue', text=energia, textposition='outside'))  

        fig.update_layout(title="Energía generada por mes", xaxis_title="Mes", yaxis_title="Energía [kWh]",
            xaxis=dict(tickangle=-45), template="seaborn")  # opciones: 'plotly' o 'seaborn', 'plotly_dark'
        st.plotly_chart(fig, use_container_width=True)

    def graficar_meses(self, tupla_meses):
        """Recibe en tupla_meses una tupla que puede señalar a uno o más meses; por ej.
        (1, 3, 4, 7), para señalar Enero, Marzo, Abril y Julio.
        Genera un par de gráficos, uno al lado del otro y en la misma ventana.
        El de la izquierda contiene las curvas de la variación temporal de la
        potencia generada a lo largo de los meses indicados. A la derecha, 
        un gráfico de barras señala la producción mensual de energía en dichos meses"""
        
        # Ajustar el ancho relativo de los gráficos
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={'width_ratios': [4, 1]})
        
        for mes in tupla_meses:
            tupla1 = (1, mes, 0, 0)
            tupla2 = (dias_por_mes[mes - 1], mes, 23, 50)
            p_rango = self.pot_generada_rango(tupla1, tupla2)
            t = np.linspace(0, 1000, len(p_rango))
            ax1.plot(t, p_rango, label=f'{dic_meses[mes]}')

        ax1.set_title('Curva de potencia por mes')
        ax1.set_xlabel('Tiempo [s]')
        ax1.set_ylabel('Potencia [kW]')
        ax1.legend()
        ax1.grid(True)

        energia = [self.energia_mes(mes) for mes in tupla_meses]
        t = np.arange(len(energia))
        
        ancho_barras = 0.5  # Ajustar el ancho de las barras
        ax2.bar(t, energia, width=ancho_barras)
        
        ax2.set_ylabel('Energia [kWh]')
        ax2.set_title(f'Energía por mes')
        ax2.set_xticks(t)
        ax2.set_xticklabels([dic_meses[mes] for mes in tupla_meses], rotation=45, ha='right')
        plt.tight_layout()
        ax2.grid(True)
        st.pyplot(fig)
        
class Generador_FV_Sta_Fe(Generador_FV):  
    def __init__(self, N=12, Ppico=240, eta=0.97, kp=-0.0044, Pinv=2.5, mu=2, Gstd=1000, Tr=25):
        """
        N: Cantidad de paneles de la instalación.
        Ppico: Potencia pico de cada panel en Watt.
        eta: Rendimiento global de la instalación, por unidad.
        kp: Coef. de temperatura-potencia en p.u./Celsius.
        Pinv: Potencia nominal del inversor en kW.
        """
        tabla_anual = datos.to_numpy()  # convierto datos en un arreglo ndarray
        Generador_FV.__init__(self, tabla_anual, N, Ppico, eta, kp, Pinv, mu, Gstd, Tr)  # Llamada al constructor de la clase base
        
        self.__Gstd = Gstd  # Atributo privado
        self.__Tr = Tr      # Atributo privado 
    
    def __str__(self):
        """
        Mensaje a mostrar con print()
        """
        txt = f'GFV con {self.N} paneles de {self.Ppico} W-pico en la ciudad de Santa Fe'
        txt2 = f'\nCoef. de potencia-temperatura: {self.kp} 1/Celsius'
        txt3 = f'\nRendimiento global de la instalación: {self.eta} (por unidad)'
        txt4 = f'\nUmbral mínimo de generación: {self.mu} (%)'
        txt5 = f'\nPotencia nominal del inversor: {self.Pinv} kW'
        txt6 = f'\n(Irrad. estándar: {self.__Gstd} W/m2 ; Temp. de ref.: {self.__Tr} °C)\n'
        
        return txt + txt2 + txt3 + txt4 + txt5 + txt6

    def __repr__(self):
        return f'<Generador FV: N={self.N}, Ppico={self.Ppico}, eta={self.eta}, ' + \
            f'kp={self.kp}, Pinv={self.Pinv}, mu={self.mu}>'


if __name__ == '__main__':
    print('\nEJEMPLOS DE USO PARA LAS FUNCIONES DEL MÓDULO:\n')
    # Características del generador fotovoltaico:
    tabla_anual = datos.to_numpy().tolist()  # para hacer pruebas con la clase madre
    N = 12
    Ppico = 240
    eta = 0.97
    kp = -0.0044
    Pinv = 2.5
    
    #gen = Generador_FV(tabla_anual, N, Ppico, eta, kp, Pinv)
    gen_UTN = Generador_FV_Sta_Fe(N, Ppico, eta, kp, Pinv)
    print(gen_UTN)

    G = 1200  # Irradiancia (W/m2).
    T = 22  # Temperatura ambiente (Celsius).
    tupla1 = (24, 4, 10, 20)  # 24 de abril a las 10:20
    tupla2 = (26, 6, 13, 0)  # 26 de junio a las 13:00

    print('Instante 1:', gen_UTN.momento(tupla1))
    print('Instante 2:',  gen_UTN.momento(tupla2))

    fila = 10000  # número de fila para hacer pruebas
    dia, mes, hora, minuto = gen_UTN.calcular_fecha_desde_fila(fila)
    print(f'Fecha calculada: {dia}/{mes} {hora:02d}:{minuto:02d} para un fila: {fila}\n')

    G1, T1 = gen_UTN.irrad_temp(1, 1, 0, 0)  # 1 de enero a las 00:00 hs
    print(f'Irradiancia: {G1} W/m2; temperatura: {T1} °C; fila: {gen_UTN.calcular_fila(1, 1, 0, 0)}')
    G2, T2 = gen_UTN.irrad_temp(24, 4, 10, 20)  # 24 de abril a las 10:20 hs
    print(f'Irradiancia: {G2} W/m2; temperatura: {T2} °C; fila: {gen_UTN.calcular_fila(24, 4, 10, 20)}\n')

    rango = gen_UTN.irrad_temp_rango(tupla1, tupla2)
    print(f"Cantidad de muestras en el intervalo entre {tupla1} y {tupla2}: {len(rango)}\n")
    print(f'Primeros 10 valores: {rango[:10]}\n')  # slice

    P = gen_UTN.pot_modelo_GFV(G, T)
    print(f'Potencia generada para una G = {G:.2f} [W/m2] y T = {T:.2f} [°C]: {P:.1f} kW\n')

    P = gen_UTN.pot_generada(tupla1)
    print(f'La potencia generada el 24 de abril a las 10:20 hs fue: {P:.2f} kW\n')

    p_interval = gen_UTN.pot_generada_rango(tupla1, tupla2)
    print(f'Primeros 10 valores de potencia generada desde el 24 de abril a las 10:20 hs:')
    c = 0
    for i in p_interval[:10]:
        print(f'{i:.3f} kW')

    print('')
    for mes in dic_meses.keys():
        pot_media_por_mes = gen_UTN.pot_media_mes(mes)
        print(f'La potencia media de {dic_meses[mes]} fue: {pot_media_por_mes:.2f} kW')

    print('')
    p_media_anual = gen_UTN.pot_media_anual()
    print(f'La potencia media anual fue: {p_media_anual:.2f} kW\n')

    for mes in dic_meses.keys():
        energia_mensual = gen_UTN.energia_mes(mes)
        print(f'La energía generada en {dic_meses[mes]} fue: {energia_mensual:.2f} kWh')

    print('')
    e_anual = gen_UTN.energia_anual()
    print(f'La energía generada en todo el año fue: {e_anual:.2f} kWh\n')

    f_utilizacion = gen_UTN.factor_de_utilizacion()
    print(f'factor de utilización: {f_utilizacion:.2f}\n')

    mes, energia_maxima = gen_UTN.max_energ_mes()
    print(f'La mayor cantidad de energía producida fue: {energia_maxima:.2f} kWh, en {dic_meses[mes]}\n')

    mes, pot_maxima = gen_UTN.max_pot_mes()
    print(f'La potencia máxima fue: {pot_maxima:.2f} kWh, en {dic_meses[mes]}\n')

    # gen_UTN.graficar_pot_rango(tupla1, tupla2)

    # gen_UTN.graficar_energia_mensual()

    tupla_meses = (1, 3, 5, 8, 12)
    
    # gen_UTN.graficar_meses(tupla_meses)
