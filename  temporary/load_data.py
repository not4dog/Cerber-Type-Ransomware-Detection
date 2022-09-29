from oauth2client.service_account import ServiceAccountCredentials
import gspread
import pandas as pd
import os

scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/spreadsheets",
         "https://www.googleapis.com/auth/drive.file",
         "https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name("C:/Users/Hwang/Downloads/cerbertype-ransomeware-data-8b6f35fff4a5.json", scope)

spreadsheet_name = "Cerbertype_Ransomeware_Data"
client = gspread.authorize(creds)
spreadsheet = client.open(spreadsheet_name)

for sheet in spreadsheet.worksheets():
    sheet

new_df = pd.read_csv('C:/Users/Hwang/Desktop/for_merge/merge.csv')
val_list = new_df.values.tolist()
load_list =val_list[0]

sheet.append_row(load_list)