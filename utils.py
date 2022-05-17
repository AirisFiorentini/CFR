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
