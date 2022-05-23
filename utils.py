# from xlwt import Workbook
# from openpyxl import load_workbook, Workbook
#
#
# def save_result_to_file(result, iteration, sheet_name):
#     try:
#         wb = load_workbook('./results.xlsx')
#         sheet = wb.get_sheet_by_name(sheet_name)
#     except:
#         wb = Workbook()
#         sheet = wb.active
#         sheet.title = sheet_name
#
#     cell = sheet.cell(row=iteration, column=1)
#     cell.value = result
#
#     wb.save('./results.xlsx')

from xlwt import Workbook


def save_result_to_file(results, sheet_name):
    # Workbook is created
    wb = Workbook('./results.xls')

    # add_sheet is used to create sheet.
    sheet1 = wb.add_sheet(sheet_name)

    counter = 0
    for result in results:
        counter += 1
        sheet1.write(counter, 0, result)

    wb.save('./results.xls')

