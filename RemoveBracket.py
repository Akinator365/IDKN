import os

if __name__ == '__main__':

    REALWORLD_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'realworld')

    # 读取原始文件并去掉每行末尾的 {}
    input_file = os.path.join(REALWORLD_DATASET_PATH, "Usairport.txt")
    output_file = os.path.join(REALWORLD_DATASET_PATH, "Usairport_cleaned.txt")

    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            cleaned_line = line.rsplit(" ", 1)[0]  # 去掉最后的 " {}"
            outfile.write(cleaned_line + "\n")

    print(f"处理完成，去掉括号的内容已保存至 {output_file}")
