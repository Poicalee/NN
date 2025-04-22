import os
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
sns.set_style("whitegrid")
import numpy as np

# Architektura sieci neuronowej - definicja warstw
nn_architecture = [
    {"input_dim": 2, "output_dim": 4, "activation": "relu"},     # warstwa wejściowa -> pierwsza warstwa ukryta
    {"input_dim": 4, "output_dim": 6, "activation": "relu"},     # pierwsza -> druga warstwa ukryta
    {"input_dim": 6, "output_dim": 6, "activation": "relu"},     # druga -> trzecia warstwa ukryta
    {"input_dim": 6, "output_dim": 4, "activation": "relu"},     # trzecia -> czwarta warstwa ukryta
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},  # czwarta warstwa ukryta -> warstwa wyjściowa
]

def init_layers(nn_architecture, seed=99):
    """
    Inicjalizacja parametrów sieci neuronowej (wagi i biasy)
    :param nn_architecture: lista słowników z definicją architektury sieci
    :param seed: ziarno dla generatora liczb losowych
    :return: słownik z inicjalizowanymi parametrami
    """
    np.random.seed(seed)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        # Inicjalizacja wag małymi losowymi wartościami
        params_values['W' + str(layer_idx)] = np.random.randn(layer_output_size, layer_input_size) * 0.1
        # Inicjalizacja biasów małymi losowymi wartościami
        params_values['b' + str(layer_idx)] = np.random.randn(layer_output_size, 1) * 0.1

    return params_values

def sigmoid(Z):
    """
    Funkcja aktywacji sigmoid: f(x) = 1 / (1 + e^(-x))
    :param Z: wektor wejściowy
    :return: wartość funkcji sigmoid dla wejścia
    """
    return 1.0 / (1.0 + np.exp(-Z))

def relu(Z):
    """
    Funkcja aktywacji ReLU: f(x) = max(0, x)
    :param Z: wektor wejściowy
    :return: wartość funkcji ReLU dla wejścia
    """
    return np.maximum(0, Z)

def sigmoid_backward(dA, Z):
    """
    Pochodna funkcji sigmoid do propagacji wstecznej
    :param dA: pochodna kosztu względem aktywacji
    :param Z: wektor wejściowy
    :return: pochodna kosztu względem Z
    """
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    """
    Pochodna funkcji ReLU do propagacji wstecznej
    :param dA: pochodna kosztu względem aktywacji
    :param Z: wektor wejściowy
    :return: pochodna kosztu względem Z
    """
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    """
    Propagacja w przód dla pojedynczej warstwy
    :param A_prev: aktywacje z poprzedniej warstwy
    :param W_curr: wagi bieżącej warstwy
    :param b_curr: biasy bieżącej warstwy
    :param activation: funkcja aktywacji
    :return: aktywacje warstwy i wartość Z
    """
    # Obliczenie liniowej części: Z = W * A + b
    Z_curr = np.dot(W_curr, A_prev) + b_curr

    # Wybór funkcji aktywacji
    if activation == "relu":
        activation_func = relu
    elif activation == "sigmoid":
        activation_func = sigmoid
    else:
        raise Exception("Niewspierana funkcja aktywacji")

    # Zastosowanie funkcji aktywacji
    return activation_func(Z_curr), Z_curr

def full_forward_propagation(X, params_values, nn_architecture):
    """
    Pełna propagacja w przód przez całą sieć
    :param X: dane wejściowe
    :param params_values: parametry sieci (wagi i biasy)
    :param nn_architecture: architektura sieci
    :return: wynik sieci i pamięć stanów pośrednich
    """
    memory = {}
    A_curr = X

    # Iteracja przez wszystkie warstwy
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr

        # Pobranie parametrów warstwy
        activ_function_curr = layer["activation"]
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]

        # Propagacja w przód dla warstwy
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

        # Zapisanie wartości w pamięci do użycia podczas propagacji wstecznej
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr

    return A_curr, memory

def convert_prob_into_class(probs):
    """
    Konwersja prawdopodobieństwa na klasy (0 lub 1)
    :param probs: prawdopodobieństwa
    :return: klasy binarne
    """
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def get_cost_value(Y_hat, Y):
    """
    Obliczenie funkcji kosztu (entropia krzyżowa)
    :param Y_hat: przewidywane prawdopodobieństwa
    :param Y: prawdziwe etykiety
    :return: wartość kosztu
    """
    m = Y_hat.shape[1]
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)

def get_accuracy_value(Y_hat, Y):
    """
    Obliczenie dokładności klasyfikacji
    :param Y_hat: przewidywane prawdopodobieństwa
    :param Y: prawdziwe etykiety
    :return: dokładność (accuracy)
    """
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()

def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    """
    Propagacja wsteczna dla pojedynczej warstwy
    :param dA_curr: pochodna kosztu względem aktywacji bieżącej warstwy
    :param W_curr: wagi bieżącej warstwy
    :param b_curr: biasy bieżącej warstwy
    :param Z_curr: wynik liniowy bieżącej warstwy
    :param A_prev: aktywacje poprzedniej warstwy
    :param activation: funkcja aktywacji
    :return: pochodne dla następnej iteracji i aktualizacje wag/biasów
    """
    # Liczba przykładów
    m = A_prev.shape[1]

    # Wybór funkcji pochodnej dla danej aktywacji
    if activation == "relu":
        backward_activation_func = relu_backward
    elif activation == "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Niewspierana funkcja aktywacji')

    # Obliczenie pochodnej funkcji aktywacji
    dZ_curr = backward_activation_func(dA_curr, Z_curr)

    # Pochodna względem wag W
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    # Pochodna względem biasów b
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    # Pochodna względem aktywacji poprzedniej warstwy
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr

def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    """
    Pełna propagacja wsteczna przez całą sieć
    :param Y_hat: przewidywane prawdopodobieństwa
    :param Y: prawdziwe etykiety
    :param memory: pamięć wartości z propagacji w przód
    :param params_values: parametry sieci
    :param nn_architecture: architektura sieci
    :return: gradienty wszystkich parametrów
    """
    grads_values = {}

    # Liczba przykładów
    m = Y.shape[1]
    # Dostosowanie kształtu wektorów
    Y = Y.reshape(Y_hat.shape)

    # Inicjalizacja gradientu dla ostatniej warstwy
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

    # Iteracja przez warstwy od końca
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        # Numerujemy warstwy sieci od 1
        layer_idx_curr = layer_idx_prev + 1
        # Pobranie funkcji aktywacji dla bieżącej warstwy
        activ_function_curr = layer["activation"]

        dA_curr = dA_prev

        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]

        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]

        # Propagacja wsteczna dla warstwy
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

        # Zapisanie gradientów
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr

    return grads_values

def update(params_values, grads_values, nn_architecture, learning_rate):
    """
    Aktualizacja parametrów sieci
    :param params_values: aktualne parametry sieci
    :param grads_values: gradienty parametrów
    :param nn_architecture: architektura sieci
    :param learning_rate: współczynnik uczenia
    :return: zaktualizowane parametry sieci
    """
    # Iteracja przez warstwy sieci
    for layer_idx, layer in enumerate(nn_architecture, 1):
        # Aktualizacja wag
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]
        # Aktualizacja biasów
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values

def train(X, Y, nn_architecture, epochs, learning_rate, verbose=False, callback=None):
    """
    Trenowanie sieci neuronowej
    :param X: dane wejściowe
    :param Y: prawdziwe etykiety
    :param nn_architecture: architektura sieci
    :param epochs: liczba epok
    :param learning_rate: współczynnik uczenia
    :param verbose: czy wyświetlać postęp
    :param callback: funkcja callback wywoływana co 50 epok
    :return: wytrenowane parametry sieci
    """
    # Inicjalizacja parametrów sieci
    params_values = init_layers(nn_architecture, 2)
    # Inicjalizacja list przechowujących historię metryk
    cost_history = []
    accuracy_history = []

    # Wykonanie obliczeń dla kolejnych iteracji
    for i in range(epochs):
        # Propagacja w przód
        Y_hat, cache = full_forward_propagation(X, params_values, nn_architecture)

        # Obliczenie metryk i zapisanie ich w historii
        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)

        # Propagacja wsteczna - obliczenie gradientu
        grads_values = full_backward_propagation(Y_hat, Y, cache, params_values, nn_architecture)
        # Aktualizacja stanu modelu
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)

        # Co 50 epok wyświetl postęp i wywołaj callback
        if(i % 50 == 0):
            if(verbose):
                print("Iteracja: {:05} - koszt: {:.5f} - dokładność: {:.5f}".format(i, cost, accuracy))
            if(callback is not None):
                callback(i, params_values)

    return params_values

# Liczba próbek w zbiorze danych
N_SAMPLES = 1000
# Stosunek między zbiorami treningowym i testowym
TEST_SIZE = 0.1

# Generowanie danych z dwóch półksiężyców
X, y = make_moons(n_samples=N_SAMPLES, noise=0.2, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

# Funkcja tworząca wykres zbioru danych
def make_plot(X, y, plot_name, file_name=None, XX=None, YY=None, preds=None, dark=False):
    """
    Tworzenie wykresu danych i granic decyzyjnych
    :param X: dane wejściowe
    :param y: etykiety
    :param plot_name: nazwa wykresu
    :param file_name: nazwa pliku do zapisu (opcjonalnie)
    :param XX: siatka X do wizualizacji granic
    :param YY: siatka Y do wizualizacji granic
    :param preds: przewidywania dla siatki
    :param dark: czy użyć ciemnego tła
    """
    if (dark):
        plt.style.use('dark_background')
    else:
        sns.set_style("whitegrid")
    plt.figure(figsize=(16,12))
    axes = plt.gca()
    axes.set(xlabel="$X_1$", ylabel="$X_2$")
    plt.title(plot_name, fontsize=30)
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)
    if(XX is not None and YY is not None and preds is not None):
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha=1, cmap=cm.Spectral)
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5], cmap="Greys", vmin=0, vmax=.6)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral, edgecolors='black')
    if(file_name):
        plt.savefig(file_name)
        plt.close()
    else:
        plt.show()

# Wyświetlenie oryginalnego zbioru danych
make_plot(X, y, "Zbiór danych")

# Granice wykresu
GRID_X_START = -1.5
GRID_X_END = 2.5
GRID_Y_START = -1.0
GRID_Y_END = 2
# Katalog wyjściowy (folder musi istnieć na dysku)
OUTPUT_DIR = "output"  # Zmieniono na domyślną nazwę folderu

# Tworzenie katalogu wyjściowego, jeśli nie istnieje
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Tworzenie siatki do wizualizacji granic decyzyjnych
grid = np.mgrid[GRID_X_START:GRID_X_END:100j, GRID_Y_START:GRID_Y_END:100j]
grid_2d = grid.reshape(2, -1).T
XX, YY = grid

def callback_numpy_plot(index, params):
    """
    Funkcja callback do generowania wykresów podczas treningu
    :param index: numer iteracji
    :param params: aktualne parametry modelu
    """
    plot_title = "Model NumPy - Iteracja: {:05}".format(index)
    file_name = "numpy_model_{:05}.png".format(index//50)
    file_path = os.path.join(OUTPUT_DIR, file_name)
    prediction_probs, _ = full_forward_propagation(np.transpose(grid_2d), params, nn_architecture)
    prediction_probs = prediction_probs.reshape(prediction_probs.shape[1], 1)
    make_plot(X_test, y_test, plot_title, file_name=file_path, XX=XX, YY=YY, preds=prediction_probs, dark=True)

# Trenowanie modelu
print("Rozpoczęcie treningu...")
params_values = train(np.transpose(X_train), np.transpose(y_train.reshape((y_train.shape[0], 1))), nn_architecture,
                      10000, 0.01, True, callback_numpy_plot)
print("Trening zakończony!")

# Testowanie modelu na siatce do wizualizacji
prediction_probs_numpy, _ = full_forward_propagation(np.transpose(grid_2d), params_values, nn_architecture)
prediction_probs_numpy = prediction_probs_numpy.reshape(prediction_probs_numpy.shape[1], 1)

# Tworzenie końcowego wykresu
make_plot(X_test, y_test, "Model NumPy - Wynik końcowy",
          file_name=os.path.join(OUTPUT_DIR, "final_result.png"),
          XX=XX, YY=YY, preds=prediction_probs_numpy)

# Obliczenie dokładności na zbiorze testowym
Y_test_hat, _ = full_forward_propagation(np.transpose(X_test), params_values, nn_architecture)
acc_test = get_accuracy_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
print(f"Dokładność na zbiorze testowym: {acc_test:.4f}")

print(f"Wykresy zostały zapisane w katalogu: {os.path.abspath(OUTPUT_DIR)}")