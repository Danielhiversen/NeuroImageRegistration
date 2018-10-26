import pyexcel_xlsx
from openpyxl import load_workbook
import sqlite3

path = "/Users/leb/OneDrive - SINTEF/Prosjekter/Nevro/Brain atlas/Location and survival - survival in days.xlsx"

conn = sqlite3.connect("/Users/leb/OneDrive - SINTEF/Prosjekter/Nevro/Brain atlas/brainSegmentation.db")
cursor2 = conn.cursor()

case_list = load_workbook(path,data_only=True)['Location and survival - surviva']
for case in range(2, 213):
    
    cell_name = "{}{}".format("A", case)
    pid = case_list[cell_name].value

    cell_name = "{}{}".format("B", case)
    survival_days = case_list[cell_name].value

    cursor2.execute("SELECT survival_days FROM Patient WHERE pid = ?", (pid,))
    survival_days_db = cursor2.fetchone()
    if survival_days_db and survival_days_db[0]:
        if survival_days_db[0] != survival_days:
            print("survival_days_db is not equal to survival_days for pid " + str(pid))
    elif survival_days != "#NULL!":
        print("Survival days for pid " + str(pid) + ": " + str(survival_days))
        cursor2.execute("UPDATE Patient SET survival_days = ? WHERE pid = ?",
                              (survival_days, pid))
        conn.commit()

cursor2.close()
conn.close()
