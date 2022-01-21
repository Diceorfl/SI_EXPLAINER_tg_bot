from clustering_interpretation import ClusteringInterpretation
from aiogram import Bot, Dispatcher, executor, types
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from config import TOKEN, HELP
from report import InterpretationReport
from io import BytesIO
import pandas as pd
import numpy as np
import logging
import json
import re

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

distribute_features_msg = "Distribute the features as follows:\n\nContinuous: feature1, feature2;\n" \
                          "Categorical: feature3, feature4;\nClusters: feature5;"

df = pd.DataFrame()
user_answers, continuous, categorical = [], [], []
interpretation_df, clusters, file_path = None, None, None
significant_features = {}

buffer = BytesIO()
interpretation_report = InterpretationReport(buffer)


def clear():
    global df, interpretation_df, user_answers, continuous,\
           categorical, clusters, file_path, buffer, interpretation_report, significant_features
    df = pd.DataFrame()
    buffer = BytesIO()
    interpretation_report = InterpretationReport(buffer)
    user_answers, continuous, categorical = [], [], []
    interpretation_df, clusters, file_path = None, None, None
    significant_features = {}


def keyboard():
    key_board = types.ReplyKeyboardMarkup(resize_keyboard=True)
    buttons = ["Yes", "No"]
    key_board.add(*buttons)
    return key_board


def define_clusters_number(x: pd.DataFrame) -> int:
    gm_bic = []
    max_bic = 0
    max_bic_pos = 0
    clusters_number = 1
    while True:
        gm = GaussianMixture(n_components=clusters_number).fit(x)
        gm_bic.append(-gm.bic(x))
        if max_bic < gm_bic[-1]:
            max_bic = gm_bic[-1]
            max_bic_pos = len(gm_bic)
        if len(gm_bic) - max_bic_pos >= 3:
            return clusters_number
        clusters_number += 1


def clustering():
    global df, clusters
    x = MinMaxScaler().fit_transform(df)
    clusters_number = define_clusters_number(x)
    model = GaussianMixture(n_components=clusters_number).fit(x)
    df["Clusters"] = model.predict(x)
    clusters = "Clusters"


def process_user_answer():
    global df, user_answers, continuous, categorical, clusters
    if not user_answers:
        raise ValueError("user answer is empty!")

    user_answer = user_answers[0].replace(" ", "")
    user_answer = user_answer.replace("\n", "")
    user_answer = re.split(':|;', user_answer)

    for i in range(0, len(user_answer) - 1):
        if user_answer[i] == "Continuous":
            continuous = list(filter(None, user_answer[i + 1].split(",")))
        if user_answer[i] == "Categorical":
            categorical = list(filter(None, user_answer[i + 1].split(",")))
        if user_answer[i] == "Clusters" and (clusters is None or not clusters):
            clusters = user_answer[i + 1]

    if not clusters or clusters is None:
        clustering()


def save_user_answers():
    global interpretation_df, user_answers
    answer = True if user_answers[-1] == "Yes" else False
    interpretation_df.set_dependent_clusters(answer)


def interpretation():
    global interpretation_df, significant_features
    significant_features = interpretation_df.get_significant_features()
    output = [["Clusters", "Significant features"]]
    for cluster in significant_features.keys():
        output.append([str(cluster)] + [", ".join(significant_features[cluster])])

    return output


def file2df(file: BytesIO):
    """
    Функция преобразует файлы типа json или csv в датафрейм.
    :param file: BytesIO
    """

    global df
    if file_path[-4:] == "json":
        df = json.load(file)
        df = pd.DataFrame.from_dict(df)
        df = pd.concat([df.apply(pd.Series)], axis=1)
    elif file_path[-3:] == "csv":
        df = pd.read_csv(file)
    else:
        raise TypeError("Неверное расширение файла!")  # если файл с каким-то другим расширением


def find_categorical() -> dict:
    categorical_features = {feature: False for feature in df.columns}
    for feature in df.columns:
        unique_percent = df[feature].nunique() / len(df[feature])
        if unique_percent <= 0.1:
            categorical_features[feature] = True

    return categorical_features


def distribute_features_automatically() -> str:
    categorical_features = find_categorical()
    cat_features = []
    noncat_features = []
    for feature in categorical_features.keys():
        if categorical_features[feature]:
            cat_features.append(feature)
        else:
            noncat_features.append(feature)

    output = "Continuous: " + ", ".join(noncat_features) + ";\n" +\
             "Categorical: " + ", ".join(cat_features[:-1]) + ";\n" +\
             "Clusters: " + cat_features[-1] + ";"

    return output


@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.reply("Hey! I'll help you figure out why patients are grouped the way they are, "
                        "and I'll also give you some information about these groups. "
                        "You can find out more information by typing the /help command.")


@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    await bot.send_document(message.from_user.id, HELP,
                            reply_to_message_id=message.message_id)


@dp.message_handler(commands=['why'])
async def process_why_command(message: types.Message):
    if interpretation_df is None:
        await message.reply("First, send the file for  interpretation!")
        return

    text = str(message.text).split("\n")

    for line in text[1:]:
        line = line.replace(" ", "")
        non = ""

        if "notin" in line:
            line = line.split("notin")
            non = "not"
        elif "in" in line:
            line = line.split("in")
        else:
            await message.reply("Example: some_id in some_cluster / some_id not in some_cluster")
            return

        try:
            id = int(line[0]) - 2
            cluster = str(line[1])
        except Exception as e:
            await message.reply("Error entering patient id or cluster!")
            return

        clstrs = list(df[clusters])
        id_clsr = list(df.iloc[[id]][clusters])

        if id not in df.index or cluster not in clstrs or (non == "" and cluster not in id_clsr)\
                or (non == "not" and cluster in id_clsr):
            await message.reply("Error entering patient id or cluster!")
            return

        msg = "Patient with id: %s is %s in the cluster: %s "\
              "because his corresponding feature values are %s similar to the mean values " \
              "of the same features in patients from this cluster:\n\n" % (str(id+2), non, str(cluster), non)
        for feature in significant_features[cluster]:
            if feature in continuous:
                msg += feature + ": patient (" + str(float(df.iloc[[id]][feature])) + ") / "
                msg += " cluster (" + str(np.mean(df[df[clusters] == cluster][feature])) + ")\n"
            if feature in categorical:
                msg += feature + ": patient (" + str(df.iloc[[id]][feature].values[0]) + ") / "
                msg += " cluster (" + str(df[df[clusters] == cluster][feature].mode()[0]) + ")\n"

        await message.answer(msg)


@dp.message_handler(commands=['getfile'])
async def process_help_command(message: types.Message):
    global df, file_path

    if file_path is None:
        await message.reply("First, send the file for interpretation!")
        return

    document = tuple()

    if file_path[-4:] == "json":
        document = "result.json", df.to_json(orient="records").encode()
    elif file_path[-3:] == "csv":
        document = "result.csv", df.to_csv(index=False).encode()

    await bot.send_document(message.from_user.id, document)


@dp.message_handler()
async def process_text_command(message: types.Message):
    global user_answers, df, interpretation_df
    if not user_answers:
        user_answers.append(str(message.text))
    if (len(user_answers) == 1 and str(message.text) == "Yes") or (user_answers[0] == str(message.text)):
        process_user_answer()
        df[clusters] = df[clusters].astype('str')
        interpretation_df = ClusteringInterpretation(df, clusters)
        interpretation_df.set_continuous_and_categorical(continuous, categorical)
        user_answers.append("Yes")
        await message.answer("Whether the compared groups are dependent?\n\n"
                             "For example, if any changes were made in the same patients repeatedly. "
                             "(Blood sugar/weight/blood pressure)",
                             reply_markup=keyboard())
        return
    if len(user_answers) == 1 and str(message.text) == "No":
        features = ", ".join(list(df.columns))
        user_answers = []
        await message.answer(distribute_features_msg)
        await message.answer(features)
    if len(user_answers) == 2:
        user_answers.append(str(message.text))
        save_user_answers()
        interpretation_report.set_table_lines(interpretation())
        interpretation_report.set_data(interpretation_df._df)
        interpretation_report.set_features(continuous=continuous, categorical=categorical)
        interpretation_report.set_differences(interpretation_df.get_differences())
        interpretation_report.create_report()
        user_id = message.from_user.id
        await bot.send_document(user_id, ("report.pdf", buffer.getvalue()))


@dp.message_handler(content_types=['document'])
async def process_document_command(message: types.Message):
    clear()

    global file_path, interpretation_report
    document_id = message.document.file_id
    file_info = await bot.get_file(document_id)
    file_path = file_info.file_path
    file = await bot.download_file(file_path)

    interpretation_report.set_file_name(message.document.file_name)

    try:
        file2df(file)
    except Exception as e:
        await message.reply("Файл содержит некоректные данные или"
                            " имеет формат отличный от json и csv!")
        return

    global user_answers
    user_answers.append(distribute_features_automatically())
    await message.answer(user_answers[-1])
    await message.answer("Has the bot distributed the features correctly?", reply_markup=keyboard())


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
