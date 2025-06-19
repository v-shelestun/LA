import os
from pathlib import Path
import pandas as pd
from collections import Counter, defaultdict
import random
import time
from threading import Timer
import json
import numpy as np  # Добавьте в самые первые строки файла
from collections import Counter
class LotteryProcessor:
    def __init__(self):
        self.data_dir = Path(r'E:\Ryad_01')
        self.top_frequent = []
        self.top_rare = []
        self.other_numbers = []
        self.current_draw = None
        self.user_input_received = False
        os.makedirs(self.data_dir, exist_ok=True)
        self.config = self.load_config()
        self.df = None 

    def find_latest_draw_file(self):
        """Поиск последнего файла с тиражами"""
        try:
            csv_files = [f for f in self.data_dir.glob('*.csv') 
                       if 'HG_4' not in f.name and 'Статистика' not in f.name]
            return max(csv_files, key=lambda x: x.stat().st_mtime) if csv_files else None
        except Exception as e:
            print(f"Ошибка поиска файлов: {str(e)}")
            return None

    def load_data(self, filepath):
        """Загрузка данных"""
        try:
            df = pd.read_csv(filepath, header=None, names=['Тираж', 'Дата', 'Числа'])
            df['Числа'] = df['Числа'].str.replace('"', '').str.strip()
            
            numbers = df['Числа'].str.split(',', expand=True)
            df[['Число1', 'Число2', 'Число3', 'Число4']] = numbers.iloc[:, :4].apply(
                lambda x: pd.to_numeric(x, errors='coerce'))
            
            df = df.dropna()
            df['Тираж'] = df['Тираж'].astype(int)
            
            if not df.empty:
                self.current_draw = df['Тираж'].iloc[0]
                return df
            return None
        except Exception as e:
            print(f"Ошибка загрузки данных: {str(e)}")
            return None
    def load_config(self):
        default_config = {
            "prediction_methods": {
                "frequency": {
                    "enabled": True,
                    "weight": 0.2,
                    "params": {"last_n": 20}
                },
                "periodicity": {
                    "enabled": True,
                    "weight": 0.8,
                    "params": {
                        "rare_symbol": "Р",
                        "max_period": 20
                    }
                },
                "pairs": {
                    "enabled": True,
                    "weight": 0.3,
                    "params": {
                        "last_n": 50,
                        "pair_types": ["ЧО", "ОЧ", "ЧР", "РЧ", "ОР", "РО", "ОО"]
                    }
                }
            }
        }

        try:
            with open(self.data_dir / 'config.json', 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                
                # Автоматическое восстановление отсутствующих ключей
                merged_config = self._deep_update(default_config, user_config)
                return merged_config
                
        except Exception as e:
            print(f"Ошибка загрузки конфига: {e}. Используются настройки по умолчанию")
            return default_config
    def generate_config(self):
        """Создает новый config.json с правильной структурой"""
        config_path = self.data_dir / 'config.json'
        default_config = {
            "prediction_methods": {
                "frequency": {
                    "enabled": True,
                    "weight": 0.2,
                    "params": {"last_n": 20}
                },
                "periodicity": {
                    "enabled": True,
                    "weight": 0.8,
                    "params": {
                        "rare_symbol": "Р",
                        "max_period": 20
                    }
                },
                "pairs": {
                    "enabled": True,
                    "weight": 0.3,
                    "params": {
                        "last_n": 50,
                        "pair_types": ["ЧО", "ОЧ", "ЧР", "РЧ", "ОР", "РО", "ОО"]
                    }
                }
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        print(f"Создан новый config.json по пути: {config_path}")
    def _deep_update(self, default, user):
        """Рекурсивное обновление конфига"""
        for key, value in user.items():
            if isinstance(value, dict) and key in default:
                default[key] = self._deep_update(default[key], value)
            else:
                default[key] = value
        return default
    def analyze_numbers(self, df, last_n=30):
        """Анализ частотности чисел (только последние N тиражей)"""
        # Берем только последние N строк
        recent_df = df.tail(last_n)
        
        all_numbers = recent_df[['Число1', 'Число2', 'Число3', 'Число4']].values.flatten()
        number_counts = Counter(all_numbers)
        
        self.top_frequent = [int(num) for num, _ in number_counts.most_common(8)]
        self.top_rare = [int(num) for num, _ in number_counts.most_common()[-8:]]
        self.other_numbers = [int(num) for num in number_counts 
                            if num not in self.top_frequent and num not in self.top_rare]

        # Вывод результатов
        print("\n" + "="*50)
        print(f" РЕЗУЛЬТАТЫ АНАЛИЗА ПОСЛЕДНИХ {last_n} ТИРАЖЕЙ ".center(50, "="))
        print(f"\nТоп-8 частых чисел: {self.top_frequent}")
        print(f"Топ-8 редких чисел: {self.top_rare}")
        print(f"\nОстальные числа ({len(self.other_numbers)}):")
        print(", ".join(map(str, self.other_numbers)))
        print("="*50 + "\n")

    def analyze_row(self, row):
        """Анализ строки"""
        result = []
        for num in row:
            if num in self.top_frequent:
                result.append(f'Ч{self.top_frequent.index(num)+1}')
            elif num in self.top_rare:
                result.append(f'Р{self.top_rare.index(num)+1}')
            else:
                if num not in self.other_numbers:
                    self.other_numbers.append(num)
                result.append(f'О{self.other_numbers.index(num)+1}')
        return ' '.join(result)

    def generate_combinations(self, formula, use_top4=False):
        """Генерация комбинаций чисел"""
        frequent = self.top_frequent[:4] if use_top4 else self.top_frequent
        rare = self.top_rare if self.top_rare else list(set(range(1, 27)) - set(self.top_frequent))
        other = self.other_numbers if self.other_numbers else list(set(range(1, 27)) - set(self.top_frequent + self.top_rare))
        
        combos = []
        for _ in range(20):
            combo = []
            for symbol in formula:
                if symbol == 'Ч':
                    pool = [n for n in frequent if n not in combo]
                    num = random.choice(pool) if pool else random.choice(frequent)
                elif symbol == 'Р':
                    pool = [n for n in rare if n not in combo]
                    num = random.choice(pool) if pool else random.choice(rare)
                else:
                    pool = [n for n in other if n not in combo]
                    num = random.choice(pool) if pool else random.choice(other)
                combo.append(int(num))
            combos.append(combo)        
        return combos

    def compare_formulas(self, pred, fact):
        """Сравнение формул по парам"""
        pred_pairs = [pred[0]+pred[1], pred[2]+pred[3]]
        fact_pairs = [fact[0]+fact[1], fact[2]+fact[3]]
        
        # Нормализация порядка в парах
        for i in range(2):
            if pred_pairs[i] in {'ОР', 'РО'}: pred_pairs[i] = 'ОР'
            elif pred_pairs[i] in {'ЧР', 'РЧ'}: pred_pairs[i] = 'ЧР'
            elif pred_pairs[i] in {'ЧО', 'ОЧ'}: pred_pairs[i] = 'ЧО'
            
            if fact_pairs[i] in {'ОР', 'РО'}: fact_pairs[i] = 'ОР'
            elif fact_pairs[i] in {'ЧР', 'РЧ'}: fact_pairs[i] = 'ЧР'
            elif fact_pairs[i] in {'ЧО', 'ОЧ'}: fact_pairs[i] = 'ЧО'
        
        match = 0
        for p, f in zip(pred_pairs, fact_pairs):
            if p == f:
                match += 1
        return match / 2 * 100

    def compare_numbers(self, pred, fact):
        """Сравнение чисел по парам"""
        pred_pairs = [set(pred[:2]), set(pred[2:])]
        fact_pairs = [set(fact[:2]), set(fact[2:])]
        matches = 0
        for p, f in zip(pred_pairs, fact_pairs):
            matches += len(p & f)
        return matches

    def update_statistics(self, pred_df=None, actual_df=None):
        """Обновляет статистику в заданном формате с сохранением данных"""
        def calculate_formula_match(pred, actual):
            """Расчет % совпадения формул с учетом тождественных пар"""
            if len(pred) != 4 or len(actual) != 4:
                return 0
            
            # Разбиваем на пары и нормализуем порядок (Ч О → О Ч и т.д.)
            def normalize_pair(pair):
                a, b = pair
                if (a == 'Ч' and b == 'О') or (a == 'О' and b == 'Ч'):
                    return ('Ч', 'О')
                elif (a == 'Р' and b == 'О') or (a == 'О' and b == 'Р'):
                    return ('Р', 'О')
                elif (a == 'Ч' and b == 'Р') or (a == 'Р' and b == 'Ч'):
                    return ('Ч', 'Р')
                return (a, b)
            
            pred_p1 = normalize_pair(pred[:2])
            pred_p2 = normalize_pair(pred[2:])
            actual_p1 = normalize_pair(actual[:2])
            actual_p2 = normalize_pair(actual[2:])
            
            # Сравниваем нормализованные пары
            full_match_first = pred_p1 == actual_p1
            full_match_second = pred_p2 == actual_p2
            
            # Совпадения символов (без учета порядка)
            first_pair_any = len(set(pred_p1) & set(actual_p1))
            second_pair_any = len(set(pred_p2) & set(actual_p2))
            
            # Определяем процент
            if full_match_first and full_match_second:
                return 100
            elif full_match_first or full_match_second:
                if (full_match_first and second_pair_any >= 1) or (full_match_second and first_pair_any >= 1):
                    return 75
                else:
                    return 50
            elif first_pair_any >= 1 and second_pair_any >= 1:
                return 50
            elif first_pair_any >= 1 or second_pair_any >= 1:
                return 25
            else:
                return 0

        def calculate_number_matches(pred, actual):
            """Расчет совпадений чисел по парам"""
            pred_p1, pred_p2 = set(pred[:2]), set(pred[2:])
            actual_p1, actual_p2 = set(actual[:2]), set(actual[2:])
            return len(pred_p1 & actual_p1) + len(pred_p2 & actual_p2)

        try:
            print("\n=== ОБНОВЛЕНИЕ СТАТИСТИКИ ===")
            
            # 1. Загрузка существующей статистики
            stats_file = self.data_dir / 'Статистика_совпадений.xlsx'
            required_columns = ['Тираж', 'Ф прогноз', 'Ф факт', '%', 'Ч прогноз', 'Ч факт', 'Совпадений']
            
            if stats_file.exists():
                existing_stats = pd.read_excel(stats_file, dtype={'Тираж': str})
                # Проверяем и приводим к нужным колонкам
                existing_stats = existing_stats[required_columns] if all(col in existing_stats.columns for col in required_columns) else pd.DataFrame(columns=required_columns)
            else:
                existing_stats = pd.DataFrame(columns=required_columns)

            # 2. Загрузка новых данных
            if pred_df is None:
                pred_file = self.data_dir / 'HG_4.csv'
                pred_df = pd.read_csv(pred_file, sep='\t', dtype={'Тираж': str}) if pred_file.exists() else pd.DataFrame()

            if actual_df is None:
                actual_df = self.load_draws()

            # 3. Обработка данных
            new_data = []
            tirages_info = {}

            for _, pred_row in pred_df.iterrows():
                try:
                    draw_id = str(pred_row['Тираж']).strip()
                    if not draw_id:
                        continue

                    # Проверяем, есть ли уже этот тираж в существующих данных
                    if not existing_stats[existing_stats['Тираж'] == draw_id].empty:
                        continue

                    # Получаем фактические данные
                    actual_data = actual_df[actual_df['Тираж'].astype(str) == draw_id]
                    if actual_data.empty:
                        continue

                    actual_row = actual_data.iloc[0]
                    
                    # Получаем формулы
                    try:
                        pred_formula = pred_row['Формула'].split()
                        actual_formula = [s[0] for s in actual_row['Анализ'].split()] if 'Анализ' in actual_row else []
                        if not actual_formula:
                            # print(f"В тираже {draw_id} отсутствуют данные анализа")
                            continue
                    except Exception as e:
                        print(f"Ошибка обработки формул для тиража {draw_id}: {str(e)}")
                        continue
                    
                    # Получаем числа
                    pred_numbers = [int(pred_row[f'Число{i}']) for i in range(1,5)]
                    actual_numbers = [int(actual_row[f'Число{i}']) for i in range(1,5)]

                    # Расчет показателей
                    formula_percent = calculate_formula_match(pred_formula, actual_formula)
                    matches = calculate_number_matches(pred_numbers, actual_numbers)

                    # Сохраняем данные прогноза
                    new_data.append({
                        'Тираж': draw_id,
                        'Ф прогноз': ' '.join(pred_formula),
                        'Ф факт': ' '.join(actual_formula),
                        '%': formula_percent,
                        'Ч прогноз': ' '.join(map(str, pred_numbers)),
                        'Ч факт': ' '.join(map(str, actual_numbers)),
                        'Совпадений': matches
                    })

                    # Аккумулируем данные для итоговой строки
                    if draw_id not in tirages_info:
                        tirages_info[draw_id] = {
                            'actual_formula': ' '.join(actual_formula),
                            'actual_numbers': ' '.join(map(str, actual_numbers)),
                            'total_matches': 0
                        }
                    tirages_info[draw_id]['total_matches'] += matches

                except Exception as e:
                    print(f"Ошибка обработки тиража {pred_row['Тираж']}: {str(e)}")
                    continue

            # 4. Формируем итоговые данные
            final_data = []
            
            # Сначала добавляем новые прогнозы
            final_data.extend(new_data)
            
            # Затем добавляем итоговые строки
            for draw_id, info in tirages_info.items():
                final_data.append({
                    'Тираж': draw_id,
                    'Ф прогноз': 'ИТОГО',
                    'Ф факт': info['actual_formula'],
                    '%': '',
                    'Ч прогноз': '',
                    'Ч факт': info['actual_numbers'],
                    'Совпадений': info['total_matches']
                })

            # 5. Объединяем и сохраняем
            if final_data:
                new_stats = pd.DataFrame(final_data)
                updated_stats = pd.concat([new_stats, existing_stats])
                
                # Удаляем возможные дубликаты (оставляем последнюю запись)
                updated_stats = updated_stats.drop_duplicates(
                    subset=['Тираж', 'Ф прогноз', 'Ч прогноз'], 
                    keep='last'
                )
                
                # Сортируем по тиражу (новые сверху) и типу записи (итого первым)
                updated_stats['sort_key'] = updated_stats['Ф прогноз'].apply(
                    lambda x: 0 if x == 'ИТОГО' else 1)
                updated_stats = updated_stats.sort_values(
                    ['Тираж', 'sort_key'], 
                    ascending=[False, True]
                ).drop(columns=['sort_key'])
                
                # Сохраняем в файл
                try:
                    updated_stats.to_excel(stats_file, index=False)
                    print(f"Статистика успешно обновлена. Добавлено {len(new_data)} новых прогнозов.")
                    return True
                except Exception as e:
                    print(f"Ошибка при сохранении: {str(e)}")
                    return False
            else:
                print("Нет новых данных для добавления")
                return True

        except Exception as e:
            print(f"Критическая ошибка: {str(e)}")
            return False
    def analyze_pairs(self, df):
        """Защищенный метод анализа пар"""
        try:
            # Получаем параметры с защитой по умолчанию
            params = self.config.get("prediction_methods", {}).get("pairs", {}).get("params", {})
            last_n = params.get("last_n", 50)
            pair_types = params.get("pair_types", ["ЧО", "ОЧ", "ЧР", "РЧ", "ОР", "РО", "ОО"])
            
            # Основная логика метода
            pair_counts = Counter()
            last_rows = df.tail(last_n)
            
            for _, row in last_rows.iterrows():
                symbols = row['Анализ'].split()
                pair1 = symbols[0] + symbols[1]
                pair2 = symbols[2] + symbols[3]
                
                for pair in [pair1, pair2]:
                    if pair in pair_types:
                        pair_counts[pair] += 1
                        
            return pair_counts.most_common()
            
        except Exception as e:
            print(f"Ошибка в analyze_pairs: {e}")
            return []
            
    def _convert_numbers_to_symbols(self, numbers_str):
        """Конвертация чисел в символы (1-9→Ч, 10-19→О, 20-25→Р)"""
        numbers = list(map(int, numbers_str.strip('""').split(',')))
        symbols = []
        for num in numbers:
            if 1 <= num <= 9:
                symbols.append("Ч")
            elif 10 <= num <= 19:
                symbols.append("О")
            elif 20 <= num <= 25:
                symbols.append("Р")
        return ' '.join(symbols)            
    def predict_with_pairs(self, df, last_n=20, pair_types=None):
        """Улучшенный анализ пар"""
        if 'symbols' not in df.columns:
            df['symbols'] = df['Числа'].apply(self._convert_numbers_to_symbols)
        
        try:
            # Получаем и анализируем пары
            last_sequences = df['symbols'].tail(last_n).tolist()
            pair_stats = {}
            
            for seq in last_sequences:
                symbols = seq.split()
                for i in range(len(symbols)-1):
                    pair = symbols[i] + symbols[i+1]
                    if not pair_types or pair in pair_types:
                        next_sym = symbols[i+1] if i+2 < len(symbols) else 'Ч'
                        pair_stats[pair] = pair_stats.get(pair, {'count':0, 'next':[]})
                        pair_stats[pair]['count'] += 1
                        pair_stats[pair]['next'].append(next_sym)
            
            if not pair_stats:
                return "Ч Ч Ч Ч"
            
            # Находим 3 самые частые пары
            top_pairs = sorted(pair_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:3]
            
            # Формируем прогноз на основе статистики
            forecast = []
            for pair in top_pairs:
                next_symbols = pair[1]['next']
                most_common_next = max(set(next_symbols), key=next_symbols.count)
                forecast.append(most_common_next)
            
            return ' '.join(forecast[:4]) if len(forecast) >=4 else 'Ч Ч Ч Ч'
            
        except Exception as e:
            print(f"Ошибка в predict_with_pairs: {str(e)}")
            return "Ч Ч Ч Ч"          
    # 1. Добавляем новый метод анализа периодичности
    def analyze_rare_periodicity(self, df, target_symbol='Р', max_period=20):
        """Анализ периодичности появления редкого символа"""
        period_stats = {period: 0 for period in range(1, max_period+1)}
        
        last_positions = {0: [], 1: [], 2: [], 3: []}
        
        for idx, row in df.iterrows():
            analysis = row['Анализ'].split()
            for pos in range(4):
                if analysis[pos][0] == target_symbol:
                    last_positions[pos].append(idx)
        
        for pos in range(4):
            positions = last_positions[pos]
            if len(positions) < 2:
                continue
                
            for i in range(1, len(positions)):
                period = positions[i] - positions[i-1]
                if period <= max_period:
                    period_stats[period] += 1
        
        total = sum(period_stats.values())
        if total > 0:
            period_stats = {k: v/total for k, v in period_stats.items()}
        
        return period_stats
    def calculate_real_periods(self, df, symbol='Р'):
        """Возвращает реальные периоды между появлениями символа"""
        periods = {0: [], 1: [], 2: [], 3: []}
        
        for pos in range(4):
            last_idx = None
            for idx, row in df.iterrows():
                if row['Анализ'].split()[pos][0] == symbol:
                    if last_idx is not None:
                        periods[pos].append(idx - last_idx)
                    last_idx = idx
                    
        return periods
    # 2. Добавляем метод прогноза с учетом периодичности
    def predict_with_periodicity(self, df, rare_symbol='Р', max_period=20):
        """
        Прогнозирует формулу с учетом периодичности появления редкого символа.
        Не требует numpy для базовой работы.
        """
        try:
            # Проверка данных
            if df.empty or max_period < 1:
                return "Ч Ч Ч Ч"
                
            recent_df = df.tail(max_period)
            if len(recent_df) < 3:  # Минимум 3 тиража для анализа
                return "Ч Ч Ч Ч"

            # Упрощенный расчет периодов без numpy
            period_stats = {pos: {'last_pos': None, 'periods': []} for pos in range(4)}
            
            prev_indices = {pos: None for pos in range(4)}
            for idx, row in recent_df.iterrows():
                analysis = row['Анализ'].split()
                for pos in range(4):
                    if analysis[pos][0] == rare_symbol:
                        if prev_indices[pos] is not None:
                            period_stats[pos]['periods'].append(idx - prev_indices[pos])
                        prev_indices[pos] = idx
                        period_stats[pos]['last_pos'] = idx

            # Формирование прогноза
            predicted = []
            current_index = recent_df.index[-1]
            
            for pos in range(4):
                stats = period_stats[pos]
                
                if not stats['periods']:
                    predicted.append(self._get_most_common_symbol(recent_df, pos))
                    continue
                    
                # Ручной расчет стандартного отклонения
                avg_period = sum(stats['periods']) / len(stats['periods'])
                squared_diffs = [(x - avg_period)**2 for x in stats['periods']]
                std_dev = (sum(squared_diffs)/len(stats['periods']))**0.5
                
                current_period = current_index - stats['last_pos']
                
                # Упрощенное условие
                if current_period > avg_period + (0.5 * std_dev if len(stats['periods']) > 1 else 0):
                    predicted.append(rare_symbol)
                else:
                    predicted.append(self._get_most_common_symbol(recent_df, pos))

            # Гарантия минимум одного rare_symbol
            if rare_symbol not in predicted:
                max_pos = max(
                    range(4),
                    key=lambda p: current_index - period_stats[p]['last_pos'] 
                    if period_stats[p]['last_pos'] else -1
                )
                predicted[max_pos] = rare_symbol

            return ' '.join(predicted)

        except Exception as e:
            print(f"Ошибка в predict_with_periodicity: {str(e)}")
            return "Ч Ч Ч Ч"

    def _get_most_common_symbol(self, df, position):
        """Вспомогательный метод: возвращает самый частый символ в позиции"""
        symbols = [row['Анализ'].split()[position][0] for _, row in df.iterrows()]
        return Counter(symbols).most_common(1)[0][0]
    def predict_next_formula(self, df, last_n=30):
        """Предсказание формулы с проверкой типов"""
        # Преобразуем last_n в int если нужно
        if isinstance(last_n, str):
            try:
                last_n = int(last_n)
            except ValueError:
                last_n = 30  # Значение по умолчанию при ошибке
                print(f"[WARNING] Некорректное значение last_n, используется {last_n}")

        # Проверка с преобразованным значением
        if len(df) < last_n:
            last_n = len(df)
        
        # Остальная логика метода...
        last_rows = df.tail(last_n)
        pos_patterns = {0: [], 1: [], 2: [], 3: []}
        
        for _, row in last_rows.iterrows():
            analysis = row['Анализ'].split()
            for pos in range(4):
                pos_patterns[pos].append(analysis[pos][0])
        
        predicted_formula = []
        for pos in range(4):
            counts = Counter(pos_patterns[pos])
            predicted_symbol = counts.most_common(1)[0][0]
            predicted_formula.append(predicted_symbol)
        
        return ' '.join(predicted_formula)
    def hybrid_prediction(self, df, last_formula=None):
        """Гибридный прогноз формулы"""
        try:
            # Базовый прогноз
            base = self.predict_next_formula(df).split()
            
            # Анализ пар
            pairs = self.analyze_pairs(df)
            if not pairs:
                return ' '.join(base)
            
            best_pair = pairs[0][0]
            
            # Варианты прогнозов
            variants = [
                f"{best_pair[0]} {best_pair[1]} {base[2]} {base[3]}",
                f"{base[0]} {base[1]} {best_pair[0]} {best_pair[1]}",
                self.predict_with_anomalies(df),
                self.predict_with_trends(df)
            ]
            
            # Исключаем последнюю формулу если нужно
            if last_formula:
                variants = [v for v in variants if v != last_formula]
            
            # Выбираем лучший вариант
            best_variant = max(variants, key=lambda x: sum(
                1 for _, row in df.tail(50).iterrows()
                if (row['Анализ'].split()[0][0] == x.split()[0] and
                    row['Анализ'].split()[1][0] == x.split()[1]) or
                   (row['Анализ'].split()[2][0] == x.split()[2] and
                    row['Анализ'].split()[3][0] == x.split()[3])
            ))
            
            return best_variant
            
        except Exception as e:
            print(f"Ошибка в hybrid_prediction: {str(e)}")
            return ' '.join(base) if 'base' in locals() else 'Ч Ч Ч Ч'
    # 3. Обновляем гибридный прогноз
    def hybrid_prediction_v3(self, df, last_formula=None):
        """Улучшенный прогноз с полным логированием"""
        print("\n=== DEBUG: НОВЫЙ МЕТОД hybrid_prediction_v3 БЫЛ ВЫЗВАН ===")
        config = self.config["prediction_methods"]
        
        # Инициализация лога
        log_header = "\n" + "="*50 + "\nПРОЦЕСС ПРИНЯТИЯ РЕШЕНИЯ\n" + "="*50
        method_logs = []
        variants = []
        
        # 1. Частотный метод
        if config["frequency"]["enabled"]:
            last_n = config["frequency"]["params"]["last_n"]
            freq_pred = self.predict_next_formula(df, last_n=last_n).split()
            
            freq_log = (
                f"\nМЕТОД: Частотный (Вес: {config['frequency']['weight']:.0%})\n"
                f"- Анализируемых тиражей: {last_n}\n"
                f"- Топ-символы: {self.get_symbol_frequencies(df, last_n)}\n"
                f"- Прогноз: {' '.join(freq_pred)}"
            )
            method_logs.append(freq_log)
            variants.append({
                "method": "frequency",
                "formula": freq_pred,
                "weight": config["frequency"]["weight"],
                "log": freq_log
            })

        # 2. Метод периодичности
        if config["periodicity"]["enabled"]:
            rare_sym = config["periodicity"]["params"]["rare_symbol"]
            max_per = config["periodicity"]["params"]["max_period"]
            periodic_pred = self.predict_with_periodicity(df, rare_sym, max_per).split()
            
            period_log = (
                f"\nМЕТОД: Периодичность (Вес: {config['periodicity']['weight']:.0%})\n"
                f"- Символ для анализа: {rare_sym}\n"
                f"- Макс. период: {max_per}\n"
                f"- Статистика периодов: {self.get_periodicity_stats(df, rare_sym)}\n"
                f"- Прогноз: {' '.join(periodic_pred)}"
            )
            method_logs.append(period_log)
            variants.append({
                "method": "periodicity",
                "formula": periodic_pred,
                "weight": config["periodicity"]["weight"],
                "log": period_log
            })
        # В hybrid_prediction_v3 после блока periodicity добавить:

        # 3. Метод пар
        # if config["pairs"]["enabled"]:
            # last_n = config["pairs"]["params"]["last_n"]
            # pair_types = config["pairs"]["params"]["pair_types"]
            # pairs_pred = self.predict_with_pairs(df, last_n, pair_types).split()
            
            # pairs_log = (
                # f"\nМЕТОД: Анализ пар (Вес: {config['pairs']['weight']:.0%})\n"
                # f"- Анализируемых тиражей: {last_n}\n"
                # f"- Типы пар: {pair_types}\n"
                # f"- Прогноз: {' '.join(pairs_pred)}"
            # )
            # method_logs.append(pairs_log)
            # variants.append({
                # "method": "pairs",
                # "formula": pairs_pred,
                # "weight": config["pairs"]["weight"],
                # "log": pairs_log
            # })
         # В блоке где вызывается pairs:
        if config["pairs"]["enabled"]:
            try:
                last_n = config["pairs"]["params"]["last_n"]
                pair_types = config["pairs"]["params"]["pair_types"]
                pairs_pred = self.predict_with_pairs(df, last_n, pair_types).split()
                
                pairs_log = (
                    f"\nМЕТОД: Анализ пар (Вес: {config['pairs']['weight']:.0%})\n"
                    f"- Анализируемых тиражей: {last_n}\n"
                    f"- Типы пар: {pair_types}\n"
                    f"- Прогноз: {' '.join(pairs_pred)}"
                )
                method_logs.append(pairs_log)
                variants.append({
                    "method": "pairs",
                    "formula": pairs_pred,
                    "weight": config["pairs"]["weight"],
                    "log": pairs_log
                })
            except Exception as e:
                print(f"Ошибка при обработке pairs: {str(e)}")           
        # 3. Логирование всех вариантов
        full_log = log_header + "\nДОСТУПНЫЕ ВАРИАНТЫ:" + "".join(method_logs)
        print(full_log)
        
        # 4. Выбор лучшего варианта (существующая логика)
        chosen = self._select_best_variant(variants, last_formula)
        
        # 5. Детальный отчет
        print("\n" + "="*50)
        print(f"ВЫБРАННЫЙ ПРОГНОЗ: {' '.join(chosen['formula'])}")
        print(f"Метод: {chosen['method']} (Вес: {chosen['weight']:.0%})")
        print("="*50)
        
        return " ".join(chosen["formula"])
    def get_symbol_frequencies(self, df, last_n=30):
        """Статистика появления символов"""
        recent = df.tail(last_n)
        symbols = []
        for _, row in recent.iterrows():
            analysis = row['Анализ'].split()
            symbols.extend([s[0] for s in analysis])
        
        freq = Counter(symbols)
        return dict(freq.most_common())

    def get_periodicity_stats(self, df, symbol):
        """Статистика периодов для символа"""
        positions = {0: [], 1: [], 2: [], 3: []}
        for idx, row in df.iterrows():
            analysis = row['Анализ'].split()
            for pos in range(4):
                if analysis[pos][0] == symbol:
                    positions[pos].append(idx)
        
        periods = []
        for pos in positions:
            for i in range(1, len(positions[pos])):
                periods.append(positions[pos][i] - positions[pos][i-1])
        
        return {
            'avg_period': sum(periods)/len(periods) if periods else 0,
            'max_period': max(periods) if periods else 0,
            'min_period': min(periods) if periods else 0
        }
    def evaluate_predictions(self, stats_df):
        """Анализ точности прогнозов"""
        correct = {
            'full': 0,  # Полное совпадение
            'partial': 0,  # Частичное
            'symbols': defaultdict(int)  # По символам
        }
        
        for _, row in stats_df.iterrows():
            pred = row['Ф прогноз'].split()
            actual = row['Ф факт'].split()
            
            # Проверка полного совпадения
            if pred == actual:
                correct['full'] += 1
            
            # Подсчет совпадений по позициям
            for p, a in zip(pred, actual):
                if p == a:
                    correct['partial'] += 1
                    correct['symbols'][p] += 1
        
        # Вывод статистики
        print("\n" + "="*50)
        print("АНАЛИЗ ТОЧНОСТИ ПРОГНОЗОВ")
        print(f"Полных совпадений: {correct['full']}/{len(stats_df)} ({correct['full']/len(stats_df):.1%})")
        print(f"Частичных совпадений: {correct['partial']}/{len(stats_df)*4} ({correct['partial']/(len(stats_df)*4):.1%})")
        
        print("\nТОЧНОСТЬ ПО СИМВОЛАМ:")
        for sym in ['Ч', 'О', 'Р']:
            total = sum(1 for row in stats_df['Ф прогноз'] if sym in row)
            acc = correct['symbols'].get(sym, 0)/total if total > 0 else 0
            print(f"- {sym}: {acc:.1%} (из {total} случаев)")        
    def _select_best_variant(self, variants, last_formula=None):
        if not variants:
            return {"method": "default", "formula": ["Ч"]*4, "weight": 0}
        
        # Фильтрация None-значений
        valid_variants = [v for v in variants if v["formula"] and all(x in ["Ч","О","Р"] for x in v["formula"])]
        
        if not valid_variants:
            return {"method": "default", "formula": ["Ч"]*4, "weight": 0}
        
        # Выбор по максимальному весу
        best = max(valid_variants, key=lambda x: x["weight"])
        
        # Дебаг-логи
        print("\n=== ДЕТАЛИ ВЫБОРА ===")
        for var in valid_variants:
            print(f"{var['method']}: вес {var['weight']:.0%} → {' '.join(var['formula'])}")
        print(f"ВЫБРАНО: {best['method']} (вес {best['weight']:.0%})")
        
        return best
    # 4. Добавляем адаптивный метод как альтернативу
    def adaptive_prediction(self, df):
        """Алгоритм, автоматически выбирающий стратегию"""
        last_20 = df.tail(20)
        rare_freq = last_20['Анализ'].str.contains('Р').mean()
        
        if rare_freq < 0.1:  # Если Р появляется реже 10%
            return self.predict_with_periodicity(df)
        else:
            # Используем сбалансированный подход
            base = self.predict_next_formula(df).split()
            periodic = self.predict_with_periodicity(df).split()
            return ' '.join([periodic[i] if random.random() < 0.4 else base[i] for i in range(4)])
    def hybrid_prediction(self, df, last_formula=None):
        """Гибридный прогноз формулы"""
        base = self.predict_next_formula(df).split()
        
        pairs = self.analyze_pairs(df)
        if not pairs:
            return ' '.join(base)
        
        best_pair = pairs[0][0]
        
        variants = [
            f"{best_pair[0]} {best_pair[1]} {base[2]} {base[3]}",
            f"{base[0]} {base[1]} {best_pair[0]} {best_pair[1]}"
        ]
        
        if last_formula:
            variants = [v for v in variants if v != last_formula]
        
        counts = []
        for variant in variants:
            v_parts = variant.split()
            count = sum(
                1 for _, row in df.tail(50).iterrows()
                if (row['Анализ'].split()[0][0] == v_parts[0] and
                    row['Анализ'].split()[1][0] == v_parts[1]) or
                   (row['Анализ'].split()[2][0] == v_parts[2] and
                    row['Анализ'].split()[3][0] == v_parts[3])
            )
            counts.append(count)
        
        return variants[counts.index(max(counts))] if variants else ' '.join(base)
    def hybrid_prediction_v4(self, df, last_formula=None):
        """Адаптивный прогноз с динамическими весами"""
        # 1. Анализ статистики
        actual_stats = self._get_actual_stats(df.tail(50))
        
        # 2. Генерация вариантов (проверяем на None)
        variants = {
            'frequency': self._get_frequency_prediction(df),
            'periodicity': self._get_periodicity_prediction(df),
            'pairs': self._get_pairs_prediction(df),
            'trend': self._get_trend_prediction(df)
        }
        valid_variants = {k: v for k, v in variants.items() if v is not None}
        
        # 3. Динамические веса с защитой от крайних значений
        rate_r = actual_stats.get('Р', {}).get('rate', 0.15)  # Дефолтное значение
        periodicity_weight = 0.6 * min(max(rate_r / 0.15, 0.1), 2.0)  # Ограничение 10%-200%
        
        weights = {
            'frequency': 0.2,
            'periodicity': periodicity_weight,
            'pairs': 0.3,
            'trend': 0.2
        }
        
        # 4. Нормализация и выбор лучшего варианта
        total_weight = sum(weights[k] for k in valid_variants)
        normalized_weights = {k: weights[k] / total_weight for k in valid_variants}
        best_variant = max(valid_variants.items(), key=lambda x: normalized_weights[x[0]])
        
        return best_variant[1]

    def _get_actual_stats(self, df):
        """Статистика появления символов в последних тиражах"""
        stats = {'Ч': {'count': 0}, 'О': {'count': 0}, 'Р': {'count': 0}}
        for _, row in df.iterrows():
            for symbol in row['Ф факт'].split():
                stats[symbol]['count'] += 1
        
        total = sum(v['count'] for v in stats.values())
        for symbol in stats:
            stats[symbol]['rate'] = stats[symbol]['count'] / total if total > 0 else 0
        
        return stats
    def _get_trend_prediction(self, df):
        """Анализ последних 5 тиражей"""
        last_formulas = [row['Ф факт'].split() for _, row in df.tail(5).iterrows()]
        trend = []
        for pos in range(4):
            symbols = [f[pos] for f in last_formulas]
            trend.append(Counter(symbols).most_common(1)[0][0])
        return ' '.join(trend)  
    def _get_pairs_prediction(self, df):
        """Генерация прогноза на основе парных комбинаций"""
        pairs = self.analyze_pairs(df)
        if not pairs:
            return "Ч Ч Ч Ч"
        
        best_pair = pairs[0][0]
        
        # Варианты использования лучшей пары
        variants = [
            f"{best_pair[0]} {best_pair[1]} Ч Ч",
            f"Ч Ч {best_pair[0]} {best_pair[1]}",
            f"{best_pair[0]} Ч {best_pair[1]} Ч"
        ]
        
        # Выбираем вариант с наибольшим числом "О"
        return max(variants, key=lambda x: x.count('О'))        
    def get_last_formula(self):
        """Получение последней формулы из файла прогнозов"""
        try:
            pred_file = self.data_dir / 'HG_4.csv'
            if pred_file.exists():
                pred_df = pd.read_csv(pred_file, sep='\t')
                if not pred_df.empty:
                    formula = pred_df['Формула'].iloc[0]
                    return formula
        except Exception as e:
            print(f"Ошибка при чтении HG_4.csv: {str(e)}")
        return None

    def input_timeout(self):
        """Обработка таймаута ввода"""
        if not self.user_input_received:
            print("\nВремя на ввод истекло. Применяется рекомендованная формула.")
            self.user_input_received = True

    def get_user_formula(self, predicted_formula):
        """Диалог выбора формулы с таймаутом"""
        if not hasattr(self, 'config'):
            self.config = self.load_config()
            
        # Проверяем доступность метода pairs
        if not self.config.get("prediction_methods", {}).get("pairs", {}).get("enabled", False):
            print("Метод pairs отключен в конфиге")
            return predicted_formula.split()
            
        try:
            pairs = self.analyze_pairs(self.df)
            last_formula = self.get_last_formula()
            
            print("\n" + "="*50)
            print(" Анализ данных ".center(50, '='))
            if last_formula:
                print(f"Последняя использованная формула: {last_formula}")
            
            print("\nТоп парных комбинаций:")
            for i, (pair, cnt, perc) in enumerate(pairs[:5], 1):
                print(f"{i}. {pair[0]}+{pair[1]} - {cnt} раз ({perc:.1f}%)")

            print("\n" + " Выбор формулы ".center(50, '='))
            print(f"Рекомендуемая формула: {predicted_formula}")
            
            # Запускаем таймер на 15 секунд
            timer = Timer(15, self.input_timeout)
            timer.start()
            
            try:
                choice = input("Использовать рекомендованную формулу? (да/нет): ").strip().lower()
                self.user_input_received = True
                
                if choice == 'нет':
                    while True:
                        custom_formula = input("Введите свою формулу (4 символа через пробел, например: Ч Р Ч О): ").upper()
                        parts = custom_formula.split()
                        if len(parts) == 4 and all(p in ['Ч','О','Р'] for p in parts):
                            return parts
                        print("Ошибка: формула должна содержать 4 символа (Ч, О, Р)")
                else:
                    return predicted_formula.split()
            finally:
                timer.cancel()
                if not self.user_input_received:
                    return predicted_formula.split()
                    
        except Exception as e:
            print(f"Ошибка анализа пар: {e}")
            return predicted_formula.split()

    def run(self):
        """Финальная версия с гарантированным созданием статистики"""
        try:
            print("=== Лотерейный анализатор ===")
            print("Проверка конфига:")
            print("pairs доступен:", self.config.get("prediction_methods", {}).get("pairs", {}).get("enabled", False))           
            # 1. Загрузка данных
            print("\n[1/6] Загрузка данных...")
            latest_file = self.find_latest_draw_file()
            if not latest_file:
                print("ОШИБКА: Не найдены файлы тиражей")
                return
            
            print(f"Обрабатываем файл: {latest_file.name}")
            self.df = self.load_data(latest_file)
            if self.df is None or self.df.empty:
                print("ОШИБКА: Не удалось загрузить данные")
                return
            # В методе run() после загрузки данных:

            # 3. Детальная проверка перед созданием статистики
            print("\n[3/6] Проверка данных для статистики...")
            pred_file = self.data_dir / 'HG_4.csv'
            stats_file = self.data_dir / 'Статистика_совпадений.xlsx'

            if pred_file.exists():
                pred_df = pd.read_csv(pred_file, sep='\t')
                
                # Детальная диагностика
                print(f"\nДИАГНОСТИКА:")
                print(f"Текущий тираж в данных: {self.current_draw}")
                print(f"Все тиражи в прогнозах: {pred_df['Тираж'].unique()}")
                print(f"Прогнозы для текущего тиража {self.current_draw}:")
                current_preds = pred_df[pred_df['Тираж'] == self.current_draw]
                print(current_preds)
                
                print(f"\nФактические данные для тиража {self.current_draw}:")
                current_actual = self.df[self.df['Тираж'] == self.current_draw]
                print(current_actual)
                
                if not current_preds.empty:
                    if not current_actual.empty:
                        print("\nВсе условия для статистики выполнены!")
                        print("Создаем статистику...")
                        self.update_statistics(current_preds, current_actual)
                        
                        # Проверка создания файла
                        if stats_file.exists():
                            print(f"Файл статистики успешно создан: {stats_file}")
                            try:
                                stats_df = pd.read_excel(stats_file)
                                print("\nСодержимое файла статистики:")
                                print(stats_df.head())
                            except Exception as e:
                                print(f"Ошибка чтения файла статистики: {e}")
                        else:
                            print("ОШИБКА: Файл статистики не был создан!")
                    else:
                        print("Нет фактических данных для сравнения")
                else:
                    print("Нет прогнозов для текущего тиража")
            else:
                print("Файл прогнозов HG_4.csv не найден")        
            # 2. Анализ чисел
            print("\n[2/6] Анализ данных...")
            self.analyze_numbers(self.df)
            self.df['Анализ'] = self.df.apply(
                lambda r: self.analyze_row([r['Число1'], r['Число2'], r['Число3'], r['Число4']]), 
                axis=1
            )
            
            # 3. Проверка и создание статистики
            print("\n[3/6] Создание статистики...")
            pred_file = self.data_dir / 'HG_4.csv'
            stats_file = self.data_dir / 'Статистика_совпадений.xlsx'
            
            if pred_file.exists():
                pred_df = pd.read_csv(pred_file, sep='\t')
                # Ищем прогнозы для текущего тиража (не предыдущего!)
                current_preds = pred_df[pred_df['Тираж'] == self.current_draw]
                
                if not current_preds.empty:
                    print(f"Найдены прогнозы для текущего тиража {self.current_draw}")
                    current_actual = self.df[self.df['Тираж'] == self.current_draw]
                    
                    if not current_actual.empty:
                        print("Создаем статистику...")
                        self.update_statistics(current_preds, current_actual)
                        print("Статистика успешно создана!")
                    else:
                        print(f"Нет фактических данных для тиража {self.current_draw}")
                else:
                    print(f"Нет прогнозов для текущего тиража {self.current_draw} в HG_4.csv")
            else:
                print("Файл прогнозов HG_4.csv не найден")
            
            # 4. Анализ для прогноза
            print("\n[4/6] Подготовка прогноза...")
            # predicted_formula = self.hybrid_prediction(self.df, self.get_last_formula())
            predicted_formula = self.hybrid_prediction_v3(self.df, self.get_last_formula())
            # 5. Получение формулы от пользователя
            print("\n[5/6] Получение формулы...")
            formula = self.get_user_formula(predicted_formula)
            combos = self.generate_combinations(formula)
            
            # 6. Сохранение нового прогноза
            print("\n[6/6] Сохранение прогноза...")
            new_pred_data = {
                'Тираж': [self.current_draw + 1] * len(combos),
                'Формула': [' '.join(formula)] * len(combos),
                'Позиция': range(1, len(combos) + 1),
                'Число1': [c[0] for c in combos],
                'Число2': [c[1] for c in combos],
                'Число3': [c[2] for c in combos],
                'Число4': [c[3] for c in combos]
            }
            
            pd.DataFrame(new_pred_data).to_csv(
                pred_file, mode='a', sep='\t',
                index=False, header=not pred_file.exists(),
                float_format='%.0f'
            )
            
            # Вывод результатов
            print("\n=== РЕЗУЛЬТАТЫ ===")
            print(f"Текущий тираж: {self.current_draw}")
            print(f"Следующий тираж: {self.current_draw + 1}")
            
            print("\nСгенерированные комбинации:")
            for i, combo in enumerate(combos, 1):
                print(f"{i:2d}. {combo[0]:2d} {combo[1]:2d} {combo[2]:2d} {combo[3]:2d}")
                
            print("\nСозданные файлы:")
            print(f"- HG_4.csv: {'СОЗДАН' if pred_file.exists() else 'ОШИБКА'}")
            print(f"- Статистика: {'СОЗДАНА' if stats_file.exists() else 'НЕТ ДАННЫХ'}")
            
            # Дополнительная диагностика
            if stats_file.exists():
                print("\nПроверьте файл статистики:")
                print(f"Путь: {stats_file}")
                try:
                    stats_df = pd.read_excel(stats_file)
                    print("\nСодержимое статистики:")
                    print(stats_df.head())
                except Exception as e:
                    print(f"Ошибка чтения файла статистики: {e}")
                
        except Exception as e:
            print(f"\nКРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            input("\nНажмите Enter для выхода...")

if __name__ == "__main__":
    processor = LotteryProcessor()
    processor.run()