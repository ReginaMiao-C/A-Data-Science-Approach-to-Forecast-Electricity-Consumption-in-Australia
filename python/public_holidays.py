"""
Function to generate the public holidays for the NSW based on:
New South Wales Consolidated Acts
PUBLIC HOLIDAYS ACT 2010 - SECT 4
https://classic.austlii.edu.au/au/legis/nsw/consol_act//pha2010163/s4.html
"""

from dateutil.easter import easter
import datetime


def next_weekday(date):
    weekday = date.weekday()
    if weekday == 5:
        date = date + datetime.timedelta(days=2)
    elif weekday == 6:
        date = date + datetime.timedelta(days=1)

    return date


def new_years_day(year):
    return next_weekday(datetime.date(year, month=1, day=1))


def australia_day(year):
    return next_weekday(datetime.date(year, month=1, day=26))

def anzac_day(year):

    day = datetime.date(year, month=4, day=25).weekday()

    if day == 5 or day == 6:
        return None
    else:
        return datetime.date(year, month=4, day=25)


def kings_birth_day(year):

    date = datetime.date(year, month=6, day=1)
    day = date.weekday()

    while not day == 0:
        date+= datetime.timedelta(days=1)
        day = date.weekday()

    return date + datetime.timedelta(days=7)


def labour_day(year):
    date = datetime.date(year, month=10, day=1)
    day = date.weekday()

    while not day == 0:
        date += datetime.timedelta(days=1)
        day = date.weekday()

    return date

def chistmas_boxing(year):
    date = datetime.date(year, month=12, day=25)
    if date.weekday() == 5 or date.weekday() == 6:
        return [date + datetime.timedelta(days=1), date + datetime.timedelta(days=2)]

    if date.weekday() == 4:
        return [date, date + datetime.timedelta(days=3)]
    else:
        return [date, date + datetime.timedelta(days=1)]

def easter_day(year):
    # returns Easter Sunday
    date = easter(year)
    return [date - datetime.timedelta(days=2), date + datetime.timedelta(days=1)]


def holidays(year):

    list_holidays = [
        new_years_day(year),
        australia_day(year),
        easter_day(year),
        anzac_day(year),
        chistmas_boxing(year),
        labour_day(year),
        kings_birth_day(year)
        ]

    list_holidays = [dt for sublist in list_holidays for dt in (sublist if isinstance(sublist, list) else [sublist])]
    # remove the None associated with a weekend Anzac Day
    list_holidays = [day for day in list_holidays if day is not None]

    list_holidays.sort()

    return list_holidays

for dy in holidays(2023):
    print(dy)
