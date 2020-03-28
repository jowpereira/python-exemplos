import pandas as pd
import requests
from bs4 import BeautifulSoup
import pandas
import datetime as dt


class Validation(object):
    """
    Exemplo de uso:
    '''
    https://www.forexfactory.com/calendar?day=today
    https://www.metalsmine.com/calendar?day=today

    '''
    y = Validation("https://www.energyexch.com/calendar?day=today")
    dataIni = dt.time(hour=8, minute=10)
    dataFim = dt.time(hour=10, minute=35)

    if (y.validation(dataIni, dataFim, 'Low')):
        print('não negocia, está dentro do horario de noticias e impacto X')
    else:
        print('negocia, está fora do horario de noticias')
    
    """
    def __init__(self, url="https://www.metalsmine.com/calendar?day=today"):
        self.__url = url

    def __process_time(self, intime, start, end):
        if start <= intime <= end:
            return True
        elif start > end:
            end_day = dt.time(hour=23, minute=59,
                              second=59, microsecond=999999)
            if start <= intime <= end_day:
                return True
            elif intime <= end:
                return True
        return False

    def validation(self, initDate, endDate, impact):
        for index, row in self.__prepareDate().iterrows():
            x = row.Time_Eastern
            if x == 'Tentative' or x == 'All Day':
                continue
            if self.__process_time(dt.time(hour=x.hour, minute=x.minute), dt.time(hour=initDate.hour, minute=initDate.minute), dt.time(hour=endDate.hour, minute=endDate.minute)) and self.__noticias[['Impact']].values[index] == impact:
                return True

        return False

    def __prepareDate(self):
        self.__noticias = self.__scraper()
        horasNoticia = pd.DataFrame(self.__noticias[["Time_Eastern"]])

        for index, row in horasNoticia.iterrows():
            if row.Time_Eastern != '':
                ultimaHonra = row
            else:
                horasNoticia.values[index] = ultimaHonra

            m2 = row.Time_Eastern

            if isinstance(m2, str):
                m2 = m2.replace(u'\xa0', '')
                if m2 == 'Tentative' or m2 == 'All Day':
                    continue
                m2 = m2.replace("p", " p")
                m2 = m2.replace("a", " a")
                m2 = dt.datetime.strptime(m2, '%I:%M %p')
            horasNoticia.values[index] = m2

        return horasNoticia

    def __scraper(self):
        session = requests.Session()
        session.max_redirects = 60
        page = session.get(self.__url)
        content = page.content
        soup = BeautifulSoup(content, "html.parser")
        table = soup.find_all("tr", {"class": "calendar_row"})
        forcal = []
        for item in table:
            dict = {}
            dict["Currency"] = item.find_all("td", {"class": "calendar__currency"})[0].text    # Currency
            dict["Event"] = item.find_all("td", {"class": "calendar__event"})[0].text.strip()  # Event Name
            dict["Time_Eastern"] = item.find_all("td", {"class": "calendar__time"})[0].text    # Time Eastern
            impact = item.find_all("td", {"class": "impact"})
            try:
                for icon in range(0, len(impact)):
                    dict["Impact"] = impact[icon].find_all(
                        "span")[0]['title'].split(' ', 1)[0]
            except:
                pass
            dict["Actual"] = item.find_all("td", {"class": "calendar__actual"})[0].text      # Actual Value
            dict["Forecast"] = item.find_all("td", {"class": "calendar__forecast"})[0].text  # forecasted Value
            dict["Previous"] = item.find_all("td", {"class": "calendar__previous"})[0].text  # Previous
            forcal.append(dict)
        df = pandas.DataFrame(forcal)
        return df[["Currency", "Event", "Impact", "Time_Eastern", "Actual", "Forecast", "Previous"]]
    
y = Validation("https://www.energyexch.com/calendar?day=today")
dataIni = dt.time(hour=8, minute=10)
dataFim = dt.time(hour=10, minute=35)

if (y.validation(dataIni, dataFim, 'Low')):
    print('não negocia, está dentro do horario de noticias e impacto X')
else:
    print('negocia, está fora do horario de noticias')