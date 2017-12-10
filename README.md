# Ad fontes, результати

## Опис даних
  На вхід маємо будь-яке .jpg зображення. У процесі зображення перетворюється в матрицю (розмірності [m x n]  у випадку чорно-білого зображення і [m x n x 3] у випадку RGB.
  Щоправда, автори в статті показують лише обробку чорно-білих та ще й тільки квадратних зображень! Оскільки часу було достатньо - ми зробили узагальнення і реалізували алгоритм і для *прямокутних кольорових* зображень RGB у два способи:
  - на вхід RGB зображеня, на вихід чорно-біле
  - на вхід і на вихід зображення RGB


## Результати експерименту

### Звіримо наші результати з результатами авторів
  
  Імпортуємо функції
    import sys
    import matplotlib.pyplot as plt
    sys.path.append('../')
    from compressor.core import compressor, show_img


  Функція для читання
    def compute_path(name):
        return '../img/' + name + '.jpg'



![lena.jpg оригінальне зображення (512х512)](https://www.researchgate.net/profile/Tao_Chen15/publication/3935609/figure/fig1/AS:394647298953219@1471102656485/Fig-1-8-bit-256-x-256-Grayscale-Lena-Image.ppm)

### Приклад виклику функцій
    lena_svd2 = compressor(
        compute_path('lena'),  # назва зображення
        rank=2,  # ранг
        im_type='gray',  # тип зображення, ще може бути 'rgb'
        compressor_type='SVD')  # тип компресингу
    
    lena_ssvd2 = compressor(
        compute_path('lena'),
        rank=2,
        im_type='gray',
        compressor_type='SSVD')
        
    show_img(lena_svd2)
    show_img(lena_ssvd2)


#### Порівнюємо результати для різних рангів:
![](https://d2mxuefqeaa7sj.cloudfront.net/s_D9EB5CC9F7F6A106423CF61CAEFE813C41F2B3870BC9F5E9203389CB7199515A_1512923300103_lena_grey.png)


Результати нашого дослідження ідентичні результатам, які отримали автори. 


### Для кольорових зображень 


![Оригінальне зображення (400x600)](https://d2mxuefqeaa7sj.cloudfront.net/s_3D738BB3E18313F4FE9B6C8644D7ABDA802ABBC64BD735DB1CB8B3CD7B019D21_1512916873341_image.png)

  **RGB to gray:**
  
![](https://d2mxuefqeaa7sj.cloudfront.net/s_D9EB5CC9F7F6A106423CF61CAEFE813C41F2B3870BC9F5E9203389CB7199515A_1512923336974_cat_grey.png)

  **RGB:**
![](https://d2mxuefqeaa7sj.cloudfront.net/s_D9EB5CC9F7F6A106423CF61CAEFE813C41F2B3870BC9F5E9203389CB7199515A_1512923349439_cat_rgb.png)


Останній приклад показує твердження авторів, що алгоритм SSVD гірше працює з точними (рівними, геометричними) фігурами. Вусики в кота на останньому зображенні у випадку використання SSVD мають гіршу якість, порівняно з SVD.

Подивитися детальніше на результати можна у файлах, які знаходяться в *experiment/results/*
Тест алгоритму знаходиться в файлі *experiment/experiment.ipynb*  , приклад виклику функції:

    visualize_results('lena', [2, 8, 14, 20, 30], im_type='gray', save=True, verbose=False)


    # 1й аргумент - назва зображення
    # 2. - тестові випадки (для рангу=2, рангу=8 ...)
    # 3. тип зображення, яке повертати
    # 4. чи зберігати зображення у файлі в папці experiment/results/
    # 5. чи виводити в консоль повідомлення про прогрес


**Обрахунок економії пам’яті**
Також, можна обчислити скільки пам’яті ми економимо, використовуючи цей алгоритм. Для порівняння ми використали зображення lena.jpg та метод бібліотеки NumPy, який зберігає файл у розширенні .npz; для рагну = 30, економія в пам’яті вийшла більше ніж у три рази.
*1025Кб / 301Кб =* ***3.4***

![](https://d2mxuefqeaa7sj.cloudfront.net/s_48374A0C3D8BACE2446869EC7E5D449B5E6760F6D3CA7573DAC3A81B421D8662_1512922096141_image.png)

![](https://d2mxuefqeaa7sj.cloudfront.net/s_D9EB5CC9F7F6A106423CF61CAEFE813C41F2B3870BC9F5E9203389CB7199515A_1512923366423_image.png)


Масиви можна знайти у *arrays* і за допомогою функції `np.load()` завантажити їх


## Висновки

### Чому навчились при розв’язанні цієї задачі?
  - Ми зрозуміли основу алгоритму SVD, змогли покращити його результати для зображень за допомогою його варіації - SSVD-алгоритму. 
  - Розібралися як працювати з такими Python бібліотеками, як NumPy, Matplotlib, PIL і змогли все поєднати для того щоб реалізувати ці алгоритми на практиці. 
  - Навчились правильно зчитування зображення, представляти зображення як матрицю і матрицю як зображення для візуалізації результатів. 
  - Написали конвертацію кольорового зображення в чорно-біле, а також реалізувати алгоритм для кольорових зображень, розкладаючи їх на спектри RGB.
### Що викликало найбільшу складність?
  - Зрозуміти як працює SVD і чому це працює.
  - Автори наводили як приклад квадратну чорно-білу картинку. Формули і пояснення складали відповідно до цього, що викликало труднощі при реалізації алгоритму перетасовки для прямокутного зображення. 
  - Найбільшу складність викликало узагальнення для кольорових зображень.
### Що лишилося незрозумілим?
  - “A bit allocation strategy” пункт в оригінальній статті


### **Виконали Роман Вей та Забульський Володимир**
