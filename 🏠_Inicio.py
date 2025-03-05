# abrir terminal y escribir: streamlit run main.py o ??_Inicio.py
# abrir navegador y escribir: http://localhost:8501 o http://192.168.1.21:8501
# https://docs.streamlit.io/develop/api-reference
# buscar en la barra lateral el comando que se quiere usar y copiar la Function Signature en main.py

import datetime  # paquetes de la librería estándar de Python
import io       

import streamlit as st  # paquetes de terceros
import pandas as pd

from Generador_FV import Generador_FV_Sta_Fe  # paquetes propios

st.set_page_config(
    page_title="Inicio",
    page_icon=":house:",
)

with st.sidebar:
    st.write('**CONFIGURACIÓN**')

    opcion = st.radio(
        "Seleccionar modo de configuración:",
        ("Configuración UTN Santa Fe", "Configuración Personalizada")
    )

    if opcion == "Configuración UTN Santa Fe":
        N = 12
        Ppico = 240
        kp = -0.0044
        eta = 0.97
        mu = 2
        Pinv = 2.5
        st.write(f"**Valores predefinidos:**\n\n"
                 f"- N = {N}\n"
                 f"- Ppico = {Ppico} W\n"
                 f"- kp = {kp} °C⁻¹\n"
                 f"- η = {eta}\n"
                 f"- µ = {mu} %\n"
                 f"- Pinv = {Pinv} kW")
    else:
        N = st.number_input('Cantidad de paneles', min_value=1, max_value=1000, value=12, step=1)
        Ppico = st.number_input('Pot. pico del panel (W)', min_value=50, max_value=1000, value=240, step=10)
        kp = st.number_input('Coef. de pot.-temp. (1/°C)', min_value=-0.01, max_value=0., value=-0.0044, step=0.0001, format='%0.4f')
        eta = st.number_input('Rendimiento global (p.u.)', min_value=0., max_value=1.0, value=0.97, step=0.01, format='%0.2f')
        mu = st.number_input('Umbral de generación (%)', min_value=0., max_value=100., value=2., step=0.1, format='%0.1f')
        Pinv = st.number_input('Potencia nominal del inversor (kW)', min_value=0.1, max_value=100., value=2.5, step=0.1, format='%0.1f')

tab1, tab2 = st.tabs(['📈 Carga de datos', '📊 Resultados'])

with tab1:
    """
    # Generador fotovoltaico
    ## Modelo básico para estimación de la potencia erogada
    Un generador fotovoltaico (GFV) convierte parte de la energía proveniente de la radiación solar en la forma eléctrica. La instalación se ejecuta en forma modular; una cantidad $N$ de paneles (o módulos) se vinculan a través de sus terminales de salida en una configuración mixta serie-paralelo. El conexionado *serie* se utiliza cuando se pretende incrementar la potencia de salida elevando el nivel de tensión eléctrica (diferencia de potencial total del conjunto). El conexionado *paralelo*, por su parte, se realiza cuando el incremento de potencia se logra elevando el nivel de la corriente entregada. En la práctica, un GFV puede utilizar una combinación de módulos conectados en serie, los que a su vez se vinculan en paralelo con otros conjuntos de conexionados serie.
    Existen numerosos nodelos matemáticos para representar el funcionamiento de un GFV. La configuración de las conexiones entre módulos es relevante si se pretende que el modelo obtenga la tensión y corriente de operación. En otras circunstancias, cuando interese fundamentalmente la potencia eléctrica entregada, pueden emplearse modelos simplificados. Por caso, la siguiente expresión obtiene la potencia eléctrica $P$ (en *kilo-Watt*) obtenida por un GFV, siempre que todos los módulos sean idénticos y cuando se utiliza un controlador de potencia que altera la condición de tensión de trabajo para maximizar el rendimiento.
    """
    st.latex(r'''P[kW] = N . \frac{G}{G_{Std}} . P_{pico} . [1+k_{p} . (T_c-T_r)] . \eta . 10^{-3}''')
    """
    donde:
    * $G$: Irradiancia global incidente en forma normal a los módulos fotovoltaicos, en $W/m^2$. La irradiancia mide el flujo de energía proveniente de la radiación solar (sea de forma directa o indirecta) por unidad de superficie incidente.
    * $G_{std}$: Irradiancia estándar, en $W/m^2$. Es un valor de irradiancia que utilizan los fabricantes de los módulos para referenciar ciertas características técnicas. Normalmente $G_{std} = 1000[W/m^2]$
    * $T_r$: Temperatura de referencia, en *Celsius*. Es una temperatura utilizada por los fabricantes de los módulos para referenciar ciertos parámetros que dependen de la temperatura. Normalmente $T_r = 25[°C]$  
    * $T_c$: Temperatura de la celda, en *Celsius*. Es la temperatura de los componentes semiconductores que conforman cada módulo fotovoltaico.  
    * $P_{pico}$: Potencia pico de cada módulo, en *Watt*. Se interpreta como la potencia eléctrica que entrega un módulo cuando $G$ coincida con $G_{std}$
    * $k_p$: Coeficiente de temperatura-potencia, en $°C^{-1}$. Es un parámetro negativo que refleja cómo incide la temperatura de la celda en el rendimiento del GFV. Se observa que incrementos (disminuciones) de $T_c$ producen, en consecuencia, disminuciones (incrementos) de $P$.
    * $\eta$: Rendimiento global de la instalación "por unidad" (valor ideal: 1). Se utiliza para considerar el efecto de sombras parciales sobre el GFV, suciedad sobre la superficie de los módulos y, fundamentalmente, el rendimiento del equipo *inversor*. Un *inversor* es un circuito electrónico que convierte la potencia eléctrica entregada por el GFV en formato de *corriente continua*, a la forma habitualmente utilizada en redes de transporte/distribución de *corriente alterna*. Esta conversión hace posible el acoplamiento del generador a una red eléctrica convencional. El inversor contemplado por el modelo de la ecuación también incluye un sistema de control para maximizar la potencia de salida.  
    La temperatura de la celda difiere de la temperatura ambiente $T$. En la literatura se disponen decenas de modelos matemáticos que permiten estimar $T_c$ a partir de mediciones de $T$. El modelo más sencillo, válido únicamente en ausencia de viento, indica que la relación se puede aproximar según:  
    """
    st.latex(r'''T_c = T + 0.031[°Cm^2/W].G''')
    """
    Se destaca, por otra parte, que las mediciones de irradiancia que se toman a partir de una estación meteorológica, normalmente no coinciden con $G$, puesto que se realizan sobre una superficie de prueba horizontal, y no en relación a la disposición real de los módulos. La obtención de $G$ a partir de las mediciones es compleja y depende, entre otras cosas, de las coordenadas geográficas del GFV (latitud y longitud), de la disposición espacial de los módulos (incluidas las inclinaciones), del momento preciso del análisis (año, mes, día, hora y zona horaria de implantación de la instalación), de la humedad relativa y temperatura del ambiente, y de las características de lo que se encuentra en los alrededores, en relación a su capacidad para reflejar en forma directa o difusa la radiación. No obstante, a los efectos de este ejercicio, se utilizarán mediciones de irradiancia asumiendo, por simplicidad, que sus valores corresponden a $G$.
    ## Umbral de generación
    Normalmente los equipos inversores funcionan adecuadamente siempre que la producción, en términos de potencia, supere un umbral mínimo $\mu$, habitualmente expresado en forma porcentual, en relación a la potencia nominal de la instalación fotovoltaica. Si este umbral no es superado, la instalación no entrega potencia eléctrica. Además, la potencia nunca supera el valor $P_{inv}[kW]$, nominal del inversor. Tomando esto en consideración, la potencia real $P_r$ que entrega la instalación se puede calcular como:  
    """
    st.latex(r'''P_{min}[kW] = \frac{\mu(\%)}{100} . P_{inv}''')
    st.latex(r'''P_r[kW]= \begin{cases}
        0 & \text{si } P \leq P_{min} \\
        P & \text{si } P_{min} < P \leq P_{inv} \\
        P & \text {si } P > P_{inv}
        \end{cases}
        ''')
    """   
    ## Carga de datos
    Para realizar la carga de datos, se debe subir un archivo en formato Excel con las columnas:
    * $G$: Irradiancia global en $W/m^2$
    * $T$: Temperatura ambiente en $°C$
    * $P$: Potencia en $kW$
    """
    
    arch = st.file_uploader('Cargar archivo de datos', type='xlsx')

    if arch:
        df = pd.read_excel(arch, index_col=0)
        G, T = df.columns  # Desempaquetar
        Tc = df[T] + 0.031 * df[G]
        df['Potencia (kW)'] = N * df[G]/1000 * Ppico * (1 + (Tc - 25)) * eta * 1e-3  # kW
        st.success('¡ARCHIVO CARGADO CON EXITO!')
        st.dataframe(df)
    else:
        st.warning('FALTA EL ARCHIVO DE DATOS', icon="⚠️")

with tab2:
    """
    Los valores especificados en la barra lateral llamada **CONFIGURACIÓN** se utilizan para calcular la potencia erogada por el GFV.   
    Por defecto, la cantidad de paneles, la potencia pico de cada panel, el coeficiente de temperatura-potencia y el rendimiento global aparecen con los valores $12$, $240[W]$, $-0.0044[°C^{-1}]$ y $0.97$ respectivamente, los cuales corresponden al generador fotovoltaico instalado en la UTN-FRSF, pero pueden ser modificados por el usuario.  
    Se asume que la irradiancia global es de $1000[W/m^2]$, la temperatura ambiente es de $25[°C]$ y el umbral de generación es del $2\%$ de la potencia nominal del inversor.  
    Volvemos a escribir las fórmulas para el cálculo de la potencia erogada por el GFV:
    """
    st.latex(r'''P[kW] = N . \frac{G}{G_{Std}} . P_{pico} . [1+k_{p} . (T_c-T_r)] . \eta . 10^{-3}''')
    st.latex(r'''T_c = T + 0.031[°Cm^2/W].G''')
    st.latex(r'''P_{min}[kW] = \frac{\mu(\%)}{100} . P_{inv}''')
    st.latex(r'''P_r[kW]= \begin{cases}
        0 & \text{si } P \leq P_{min} \\
        P & \text{si } P_{min} < P \leq P_{inv} \\
        P & \text {si } P > P_{inv}
        \end{cases}
        ''')

    """
    La siguiente tabla permite seleccionar un día en particular, de los datos cargados. Luego se grafican la potencia erogada, temperatura e irradiancia del GFV a lo largo de ese día.
    """
    d = st.date_input('Seleccionar día', value=datetime.date(2019, 1, 15), # buscar del archivo excel el primer y el último día
                      min_value=datetime.date(2019, 1, 1), 
                      max_value=datetime.date(2019, 12, 31), 
                      format="YYYY/MM/DD")
    if arch:
        tabla_filtrada = df.loc[f'{d.year}-{d.month}-{d.day}', :]
        st.dataframe(tabla_filtrada)

        """ ### Gráficos de potencia, temperatura e irradiancia en función de la hora del día"""
        
        st.line_chart(data=tabla_filtrada, y='Potencia (kW)', x_label='Hora', y_label='Pot. (kW)', 
                      color=None, width=None, height=None, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(data=tabla_filtrada, y=T, x_label='Hora', 
                      y_label='Temperatura ( °C)')
        with col2:
            st.line_chart(data=tabla_filtrada, y=G, x_label='Hora', 
                      y_label=G)
        nuevo_archivo = io.BytesIO()  # Crea un archivo temporal en memoria RAM
        tabla_filtrada.to_excel(nuevo_archivo)
        st.download_button('Descargar resultados', data= nuevo_archivo, 
                           file_name='Tabla_resultados.xlsx', icon='⬇️')
        
        generador = Generador_FV_Sta_Fe(N, Ppico, eta, kp, Pinv, mu)
        
        """ ### Seleccionar el rango de tiempo para graficar la potencia generada"""

        # Selección de fecha y hora inicial
        col1, col2 = st.columns(2)
        with col1:
            fecha_inicio = st.date_input("Fecha de inicio", value=datetime.date(2019, 1, 1),
                                        min_value=datetime.date(2019, 1, 1),
                                        max_value=datetime.date(2019, 12, 31))
            hora_inicio = st.time_input("Hora de inicio", value=datetime.time(0, 0))

        # Selección de fecha y hora final
        with col2:
            fecha_fin = st.date_input("Fecha de fin", value=datetime.date(2019, 1, 2),
                                    min_value=datetime.date(2019, 1, 1),
                                    max_value=datetime.date(2019, 12, 31))
            hora_fin = st.time_input("Hora de fin", value=datetime.time(23, 50))

        # Convertir selección del usuario en tuplas (d, m, h, mi)
        tupla1 = (fecha_inicio.day, fecha_inicio.month, hora_inicio.hour, hora_inicio.minute)
        tupla2 = (fecha_fin.day, fecha_fin.month, hora_fin.hour, hora_fin.minute)

        # Botón para graficar
        if st.button("Graficar potencia generada en el rango seleccionado"):
            generador.graficar_pot_rango(tupla1, tupla2)

        if st.button('Graficar energía generada en cada mes del año'):
            generador.graficar_energia_mensual()
        