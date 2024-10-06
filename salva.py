def suma_digitos(numero):
    suma = 0
    while numero > 0:
        suma += numero % 10  # Sumar el último dígito
        numero //= 10         # Eliminar el último dígito
    return suma

def procesar_numero(numero):
    iteraciones = 0
    while numero >= 10:
        suma = suma_digitos(numero)  # Calcular la suma de los dígitos

        # Si la suma de los dígitos es mayor que 9
        if suma > 9:
            if suma % 2 == 0:          # Si la suma es par
                suma *= 2
            else:                       # Si la suma es impar
                suma -= 1
        
        numero = suma                 # Asignar la nueva suma al número
        iteraciones += 1              # Contar la iteración
    
    return numero, iteraciones


numero = int(input("Ingresa un número entero positivo: "))
if numero < 1:
    print("Por favor, ingresa un número entero positivo mayor que 0.")
else:
        numero_final, cantidad_iteraciones = procesar_numero(numero)
        print("Número final:", numero_final)
        print("Cantidad de iteraciones:", cantidad_iteraciones)
