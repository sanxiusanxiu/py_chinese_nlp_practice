import re


# 代码3-17 验证电子邮件地址格式
def match_email_address(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    if re.match(pattern, email):
        return True
    else:
        return False


email1 = "test@example.com"
email2 = "invalid_email"
print(match_email_address(email1))  # 输出 True
print(match_email_address(email2))  # 输出 False


# 代码3-18 验证电话号码
def validate_phone_number(phone_number):
    pattern = r'^\d{3}-\d{8}$'
    if re.match(pattern, phone_number):
        return True
    else:
        return False


phone_number = '123-45678901'
if validate_phone_number(phone_number):
    print("电话号码格式正确")
else:
    print("电话号码格式不正确")


# 代码3-19 验证日期格式
def validate_date(date):
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    if re.match(pattern, date):
        return True
    else:
        return False


date = '2023-10-27'
if validate_date(date):
    print("日期格式正确")
else:
    print("日期格式不正确")

# 代码3-20 提取文本中姓名、性别、年龄、电话号码和地址等信息
text = """
姓名：张三
性别：男
年龄：25岁
电话号码：13512345678
地址：北京市朝阳区
"""
name_pattern = r"姓名：(.*?)\n"
gender_pattern = r"性别：(.*?)\n"
age_pattern = r"年龄：(.*?)岁\n"
phone_pattern = r"电话号码：(.*?)\n"
address_pattern = r"地址：(.*?)\n"
name = re.search(name_pattern, text).group(1)
gender = re.search(gender_pattern, text).group(1)
age = re.search(age_pattern, text).group(1)
phone = re.search(phone_pattern, text).group(1)
address = re.search(address_pattern, text).group(1)
print("姓名:", name)
print("性别:", gender)
print("年龄:", age)
print("电话号码:", phone)
print("地址:", address)
