

if __name__ == '__main__':
    # 读取原始文件并去掉每行末尾的 {}
    input_file = "D:\\lwh\IDKN\\data\\networks\\realworld\\Infectious.txt"  # 替换为你的文件名
    output_file = "D:\\lwh\IDKN\\data\\networks\\realworld\\Infectious_new.txt"

    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            cleaned_line = line.rsplit(" ", 1)[0]  # 去掉最后的 " {}"
            outfile.write(cleaned_line + "\n")

    print(f"处理完成，去掉括号的内容已保存至 {output_file}")
