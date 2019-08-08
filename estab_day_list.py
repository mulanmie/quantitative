def estab_day_list(day_begin,day_end):
    date_list = []
    year_begin = day_begin // 10000
    month_begin = day_begin // 100 % 100
    date_begin = day_begin % 100
    year_end = day_end // 10000
    month_end = day_end // 100 % 100
    date_end = day_end % 100
    if year_end > year_begin:
        interval = (year_end-year_begin)*372 + (month_end * 31 + date_end) - (month_begin * 31 + date_begin)
    else:
        interval = (month_end * 31 + date_end) - (month_begin * 31 + date_begin)
    print(interval)
    if interval < 0:
        print("cannot get the right date interval")
    else:
        date_list.append(day_begin)
        date = date_begin
        month = month_begin
        year = year_begin
        if interval != 0:
            for i in range(interval):
                if date < 31:
                    date += 1
                elif month < 12:
                    month += 1
                    date = 1
                else:
                    year += 1
                    month = 1
                    date = 1
                day = year * 10000 + month * 100 + date
                date_list.append(day)
    print(date_list)
    return date_list

if __name__ == "__main__":
    estab_day_list(20181228, 20190106)


    