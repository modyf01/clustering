import pytest
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

_QUALITY_REPORTS = []


@pytest.fixture(scope="session")
def quality_recorder():
    def record(model_name: str, num_clusters: int, silhouette: float, calinski_harabasz: float, davies_bouldin: float):
        _QUALITY_REPORTS.append(
            f"[quality] model={model_name} k={num_clusters} silhouette={silhouette:.4f} ch={calinski_harabasz:.2f} db={davies_bouldin:.4f}"
        )
    return record


@pytest.fixture(scope="session")
def quality_log():
    def log(line: str):
        _QUALITY_REPORTS.append(line)
    return log


@pytest.fixture(scope="session")
def expected_groups():
    # Cztery tematy (4 grupy), w kolejności indeksów z event_reports_noisy (PL 12, EN 12, RU 12, UK 12)
    # 0..11: PL, 12..23: EN, 24..35: RU, 36..47: UK
    # Każdy temat ma po 3 teksty w każdym języku, więc 12 indeksów na grupę
    # Grupa 0: dron (znaleziono/widziano/zestrzelono)
    drone = [
        0, 1, 2,       # PL
        12, 13, 14,    # EN
        24, 25, 26,    # RU
        36, 37, 38,    # UK
    ]
    # Grupa 1: kolejki zbyt duże na koncert
    queues = [
        3, 4, 5,       # PL
        15, 16, 17,    # EN
        27, 28, 29,    # RU
        39, 40, 41,    # UK
    ]
    # Grupa 2: toalety (zapchane/brudne/wylewa)
    toilets = [
        6, 7, 8,       # PL
        18, 19, 20,    # EN
        30, 31, 32,    # RU
        42, 43, 44,    # UK
    ]
    # Grupa 3: aplikacja biletowa (buguje/zacina/nie działa)
    ticket_app = [
        9, 10, 11,     # PL
        21, 22, 23,    # EN
        33, 34, 35,    # RU
        45, 46, 47,    # UK
    ]
    return [drone, queues, toilets, ticket_app]


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if _QUALITY_REPORTS:
        terminalreporter.write_line("\n=== Clustering quality summary ===")
        for line in _QUALITY_REPORTS:
            terminalreporter.write_line(line)


@pytest.fixture(scope="session")
def event_reports_noisy():
    """
    Multilingual (PL/EN/RU/UK) noisy reports for 4 topics with at least 3 texts
    per language per topic. Many emojis and filler words included.
    Returned: list of dicts { 'lang', 'text' }.
    Order: 12 PL, 12 EN, 12 RU, 12 UK (48 total).
    """
    data = [
        # Polish (pl) - 12 entries (3 per topic)
        # DRONE (3)
        {"lang": "pl", "text": "Widziałem, no wiecie, drona nad stadionem 😳😳, eee chyba wrogiego ✈️🚫"},
        {"lang": "pl", "text": "Tak jakby znaleziono drona przy wejściu 🤖📦, yyy ochrona zabrała 🙄"},
        {"lang": "pl", "text": "Generalnie drona zestrzelono poza sceną 💥✈️, eee wszyscy krzyczeli 😮"},
        # QUEUES (3)
        {"lang": "pl", "text": "Kolejki na koncert były za duże 😩😩, w ogóle nie da się wejść 🚪⏳"},
        {"lang": "pl", "text": "No bo serio, bardzo długie kolejki, tak jakby totalny korek ludzi 🧍🧍🧍"},
        {"lang": "pl", "text": "Kolejki były ogromne 🤯🤯, yyy czekaliśmy godzinę, eee masakra ⏱️"},
        # TOILETS (3)
        {"lang": "pl", "text": "Zapchana toaleta 🤢🚽, w ogóle śmierdzi, no i się wylewa 💦"},
        {"lang": "pl", "text": "Bardzo brudna toaleta 🤮🚽, tak jakby zero sprzątania, eee dramat"},
        {"lang": "pl", "text": "Z toalety się wylewa 😱🚽💧, generalnie nie da się wejść"},
        # TICKET APP (3)
        {"lang": "pl", "text": "Aplikacja do biletów się buguje 📱🐞, no po prostu nie działa 🙅"},
        {"lang": "pl", "text": "Aplikacja biletowa strasznie się zacina 😤📱, yyy nie mogę kupić"},
        {"lang": "pl", "text": "Aplikacja nie działa 🤯📱, tak jakby próby kończą się błędem ❌"},

        # English (en) - 12 entries
        # DRONE (3)
        {"lang": "en", "text": "We saw, like, a drone flying over the venue 😳😳, maybe hostile ✈️🚫"},
        {"lang": "en", "text": "They kinda found a drone near the gate 🤖📦, um security took it 🙄"},
        {"lang": "en", "text": "Basically they shot down a drone outside 💥✈️, people screamed 😮"},
        # QUEUES (3)
        {"lang": "en", "text": "Queues were way too long 😩😩, like we couldn't get in 🚪⏳"},
        {"lang": "en", "text": "Super long lines, kinda total human jam 🧍🧍🧍, uh stuck"},
        {"lang": "en", "text": "Lines were massive 🤯🤯, waited an hour ⏱️, basically awful"},
        # TOILETS (3)
        {"lang": "en", "text": "Clogged toilet 🤢🚽, like it stinks and overflows 💦"},
        {"lang": "en", "text": "Really dirty restroom 🤮🚽, kinda no cleaning, uh terrible"},
        {"lang": "en", "text": "The toilet is overflowing 😱🚽💧, can't even enter"},
        # TICKET APP (3)
        {"lang": "en", "text": "Ticket app is bugging out 📱🐞, basically doesn't work 🙅"},
        {"lang": "en", "text": "The ticket app freezes badly 😤📱, like can't purchase"},
        {"lang": "en", "text": "App doesn't work 🤯📱, every try ends with error ❌"},

        # Russian (ru) - 12 entries
        # DRONE (3)
        {"lang": "ru", "text": "Мы, типа, видели дрон над площадкой 😳😳, возможно вражеский ✈️🚫"},
        {"lang": "ru", "text": "Нашли дрон у входа 🤖📦, короче охрана забрала 🙄"},
        {"lang": "ru", "text": "Дрон сбили снаружи 💥✈️, народ закричал 😮"},
        # QUEUES (3)
        {"lang": "ru", "text": "Очереди слишком длинные 😩😩, вообще не попасть 🚪⏳"},
        {"lang": "ru", "text": "Очень длинные очереди, типа полный затор людей 🧍🧍🧍"},
        {"lang": "ru", "text": "Очереди огромные 🤯🤯, ждали час ⏱️, ну такое"},
        # TOILETS (3)
        {"lang": "ru", "text": "Туалет забит 🤢🚽, воняет и переливается 💦"},
        {"lang": "ru", "text": "Очень грязный туалет 🤮🚽, вообще не убирают, короче ужас"},
        {"lang": "ru", "text": "Из туалета льёт 😱🚽💧, зайти нельзя"},
        # TICKET APP (3)
        {"lang": "ru", "text": "Приложение для билетов глючит 📱🐞, вообще не работает 🙅"},
        {"lang": "ru", "text": "Приложение дико зависает 😤📱, купить нельзя"},
        {"lang": "ru", "text": "Приложение не работает 🤯📱, каждая попытка с ошибкой ❌"},

        # Ukrainian (uk) - 12 entries
        # DRONE (3)
        {"lang": "uk", "text": "Ми, типу, бачили дрона над майданчиком 😳😳, можливо ворожого ✈️🚫"},
        {"lang": "uk", "text": "Знайшли дрона біля входу 🤖📦, коротше охорона забрала 🙄"},
        {"lang": "uk", "text": "Дрона збили зовні 💥✈️, люди закричали 😮"},
        # QUEUES (3)
        {"lang": "uk", "text": "Черги надто довгі 😩😩, взагалі не зайти 🚪⏳"},
        {"lang": "uk", "text": "Дуже довгі черги, типу повний затор 🧍🧍🧍"},
        {"lang": "uk", "text": "Черги величезні 🤯🤯, чекали годину ⏱️, ну так собі"},
        # TOILETS (3)
        {"lang": "uk", "text": "Унітаз забитий 🤢🚽, смердить і переливає 💦"},
        {"lang": "uk", "text": "Дуже брудний туалет 🤮🚽, взагалі не прибирають"},
        {"lang": "uk", "text": "З туалету ллє 😱🚽💧, зайти неможливо"},
        # TICKET APP (3)
        {"lang": "uk", "text": "Додаток для квитків багається 📱🐞, взагалі не працює 🙅"},
        {"lang": "uk", "text": "Додаток жахливо зависає 😤📱, купити неможливо"},
        {"lang": "uk", "text": "Додаток не працює 🤯📱, кожна спроба з помилкою ❌"},
    ]

    return data


@pytest.fixture(scope="session")
def event_texts_list(event_reports_noisy):
    return [row["text"] for row in event_reports_noisy]


@pytest.fixture(scope="session")
def event_langs_set(event_reports_noisy):
    return {row["lang"] for row in event_reports_noisy}


