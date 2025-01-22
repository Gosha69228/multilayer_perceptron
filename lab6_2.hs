import Text.Printf
import Control.Monad (foldM)

-- Определение типа для матриц
type Matrix = [[Double]]

-- Умножение матриц
matMul :: Matrix -> Matrix -> Matrix
matMul a b = [[sum [a !! i !! k * b !! k !! j | k <- [0 .. length (head a) - 1]] | j <- [0 .. length (head b) - 1]] | i <- [0 .. length a - 1]]

-- Сложение матриц
matAdd :: Matrix -> Matrix -> Matrix
matAdd a b = [[a !! i !! j + b !! i !! j | j <- [0 .. length (head a) - 1]] | i <- [0 .. length a - 1]]

-- Умножение матрицы на скаляр
matScale :: Double -> Matrix -> Matrix
matScale s a = [[s * a !! i !! j | j <- [0 .. length (head a) - 1]] | i <- [0 .. length a - 1]]

-- Единичная матрица
identityMatrix :: Int -> Matrix
identityMatrix n = [[if i == j then 1 else 0 | j <- [0 .. n - 1]] | i <- [0 .. n - 1]]

-- Нулевая матрица
zeroMatrix :: Int -> Int -> Matrix
zeroMatrix rows cols = [[0 | _ <- [1 .. cols]] | _ <- [1 .. rows]]

-- Функция для решения уравнения методом Эйлера
eulerMethod :: (Matrix -> Matrix) -> Matrix -> Double -> [Double] -> [Matrix]
eulerMethod f y0 h xs = scanl (\y x -> matAdd y (matScale h (f y))) y0 xs

-- Правая часть уравнения
equation :: Matrix -> Matrix
equation x =
    let e = [[1, 0], [0, 0]]  -- Матрица E
        a = [[-1, 1], [0, -2]] -- Матрица A
        b = [[0], [1]]         -- Матрица B
        u = [[1]]              -- Управление u(t) = 1 (для примера)
        term1 = matMul a x     -- A * x(t)
        term2 = matMul b u     -- B * u(t)
    in matAdd term1 term2      -- A * x(t) + B * u(t)

-- Простая нейронная сеть с одним скрытым слоем
type Network = ([Double], [Double], [Double], [Double]) -- Веса и смещения

-- Ограничение значений (для стабильности)
clamp :: Double -> Double -> Double -> Double
clamp minVal maxVal x = max minVal (min maxVal x)

-- Прямое распространение (forward pass)
forward :: Network -> Double -> Double
forward (w1, b1, w2, b2) x =
    let hidden = map (\(wi, bi) -> tanh (clamp (-10) 10 (wi * x + bi))) (zip w1 b1)
        output = sum (zipWith (*) hidden w2) + b2 !! 0
    in output

-- Функция потерь (MSE)
loss :: Network -> [Double] -> [Double] -> Double
loss network xs ys =
    let predictions = map (forward network) xs
        squaredErrors = zipWith (\yPred yTrue -> (yPred - yTrue)^2) predictions ys
    in sum squaredErrors / fromIntegral (length xs)

-- Вычисление градиента для одного параметра
computeGradient :: (Network -> Double) -> Network -> Int -> Double
computeGradient lossFn network paramIndex =
    let epsilon = 1e-6
        networkPlus = updateParam network paramIndex epsilon
        networkMinus = updateParam network paramIndex (-epsilon)
    in (lossFn networkPlus - lossFn networkMinus) / (2 * epsilon)

-- Обновление параметра в сети
updateParam :: Network -> Int -> Double -> Network
updateParam (w1, b1, w2, b2) paramIndex delta =
    case paramIndex of
        0 -> (updateList w1 0 delta, b1, w2, b2)
        1 -> (w1, updateList b1 0 delta, w2, b2)
        2 -> (w1, b1, updateList w2 0 delta, b2)
        3 -> (w1, b1, w2, updateList b2 0 delta)
        _ -> error "Invalid parameter index"

-- Обновление элемента в списке
updateList :: [Double] -> Int -> Double -> [Double]
updateList xs i delta =
    let (before, x : after) = splitAt i xs
    in before ++ (x + delta : after)

-- Градиентный спуск
gradientDescent :: Network -> [Double] -> [Double] -> Double -> Network
gradientDescent network xs ys learningRate =
    let lossFn = \net -> loss net xs ys
        gradW1 = map (\i -> computeGradient lossFn network i) [0 .. length (networkW1 network) - 1]
        gradB1 = map (\i -> computeGradient lossFn network i) [0 .. length (networkB1 network) - 1]
        gradW2 = map (\i -> computeGradient lossFn network i) [0 .. length (networkW2 network) - 1]
        gradB2 = map (\i -> computeGradient lossFn network i) [0 .. length (networkB2 network) - 1]
        newW1 = zipWith (-) (networkW1 network) (map (* learningRate) gradW1)
        newB1 = zipWith (-) (networkB1 network) (map (* learningRate) gradB1)
        newW2 = zipWith (-) (networkW2 network) (map (* learningRate) gradW2)
        newB2 = zipWith (-) (networkB2 network) (map (* learningRate) gradB2)
    in (newW1, newB1, newW2, newB2)

-- Вспомогательные функции для доступа к параметрам сети
networkW1 :: Network -> [Double]
networkW1 (w1, _, _, _) = w1

networkB1 :: Network -> [Double]
networkB1 (_, b1, _, _) = b1

networkW2 :: Network -> [Double]
networkW2 (_, _, w2, _) = w2

networkB2 :: Network -> [Double]
networkB2 (_, _, _, b2) = b2

-- Обучение модели
train :: Network -> [Double] -> [Double] -> Double -> Int -> IO Network
train network xs ys learningRate epochs =
    let trainStep net epoch = do
            let newNet = gradientDescent net xs ys learningRate
                currentLoss = loss newNet xs ys
            -- Вывод информации о каждой 100-й эпохе
            if epoch `mod` 100 == 0
                then printf "Epoch %d, Loss: %.6f\n" epoch currentLoss
                else return ()
            return newNet
    in foldM trainStep network [1 .. epochs]

-- Генерация данных для обучения
xs :: [Double]
xs = [0, 0.05 .. 5]  -- Временные точки

-- Начальное состояние x(0) = [1, 1]
initialState :: Matrix
initialState = [[1], [1]]

-- Численное решение уравнения
ys :: [Matrix]
ys = eulerMethod equation initialState 0.01 xs

-- Инициализация сети
initialNetwork :: Network
initialNetwork = ([0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.01])

-- Обучение и проверка решения
main :: IO ()
main = do
    putStrLn "Решаем матричное уравнение с помощью персептрона:"
    putStrLn "E * x'(t) = A * x(t) + B * u(t)"
    -- Используем только первую компоненту вектора состояния для обучения
    let xsTrain = xs
        ysTrain = map (head . head) ys  -- Берем первую компоненту x(t)
    trainedNetwork <- train initialNetwork xsTrain ysTrain 0.001 1000
    putStrLn "\nРезультат решения уравнения:"
    mapM_ (\(t, y) -> printf "t = %.2f, x(t) = [%.6f, %.6f]\n" t (y !! 0 !! 0) (y !! 1 !! 0)) (zip xs ys)