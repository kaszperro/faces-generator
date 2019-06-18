# Faces & Flowers Generator

Aplikacja generująca twarze i kwiatki używając DCGAN. Można tego użyć do tworzenia nieprawdziwych profili w internecie (w celach oszustw), lub patrząc bardziej przyszłościowo - generować nowe twarze w kinematografii, odtwarzać starszych aktorów.

## Pierwsze kroki

Przed uruchomieniem aplikacji wymagane jest zainstalowanie wszystkich zależności używając pipenv:

```
pipenv install
```

## Instrukcja obsługi

Aby otworzyć główną aplikację w której wyświetlają się nam generowane obrazy w zależności od wprowadzonych parametrów wystarczy użyć komendy:

```
pipenv run python3 GeneratorApp.py
```

### Dodatkowe moduły

#### GifMaker

Dla zobrazowania jak postępowało uczenie sieci, z obrazów pokazujących kolejne iteracje stworzyliśmy gif'y które przechowujemy pod ściężką "./trained/{faces|flowers}/generated/gifs/".
W celu ich generacji napisaliśmy zautomatyzowany skrypt, do którego obsługi dodaliśmy proste menu konsolowe:

```
pipenv run python3 GifMaker.py
```

#### ImageComparer

Dodaliśmy możliwość wyszukiwania podobnego obrazu (twarzy/kwiatka) w wygenerowanych obrazach bazując na MSE (Mean Squared Error - Błąd średniokwadratowy) i SSIM (Structural Similarity Index). Zapisujemy łącznie 15 ostatnio odnalezionych obrazów (5 najlepszych MSE, 5 najlepszych SSIM i 5 najlepszych połączeń obu tych parametrów).
Możliwe jest wyświetlenie porównania obrazu używanego do wyszukania wraz ze znalezionymi.
Tutaj także dodaliśmy do obsługi proste menu konsolowe:

```
pipenv run python3 ImageComparer.py
```

**UWAGA - Ostatnie wyniki wyszukiwania zostaną nadpisane jeśli uruchomimy wyszukiwanie**

Pod ścieżką "./search/" znajdują się pliki powiązane z tym modułem:

* "./search/data" - Zawiera pliki tekstowe powiązane z zapisanymi obrazami, w których znajduje się wektor użyty do generacji oraz parametry MSE i SSIM
* "./search/img" - Zawiera znalezione i zapisane obrazy
