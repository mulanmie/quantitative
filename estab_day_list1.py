import os

def estab_day_list(day_begin,day_end):
    date_list = []
    date_index = []
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
    # print(len(date_list))
    data_path = '/data/remoteDir/server_200/mem_data/'
    for date in date_list:
        year = str(date // 10000).zfill(4)
        month = str(date // 100 % 100).zfill(2)
        day= str(date% 100).zfill(2)    
        path_day = os.path.join(data_path,year,month,day,'open')
        if not os.path.exists(path_day):
            print(str(date) + 'not exit')
        else:
            date_index.append(date)
    # print(len(date_index))
    return date_index

if __name__ == "__main__":
    estab_day_list(20181228, 20190106)

    