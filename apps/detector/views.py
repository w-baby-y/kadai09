import random
import cv2
import numpy as np
import torch
import torchvision

import uuid
from pathlib import Path

from apps.app import db
from apps.crud.models import User

from apps.detector.forms import UploadImageForm, DetectorForm
from apps.detector.models import UserImage, UserImageTag
from flask import (
    Blueprint,
    render_template,
    current_app,
    send_from_directory,
    redirect,
    url_for,
    flash,
)
from PIL import Image
from sqlalchemy.exc import SQLAlchemyError
from flask_login import current_user, login_required

# templateフォルダの指定
dt = Blueprint("detector", __name__, template_folder="templates")


@dt.route("/")
def index():
    # UserとUserImageをジョインして画像一覧を取得
    # ジョインしているので(<User 3>, <UserImage 1>)となる。この変数の中身を呼び出したい場合は、forで回して、user_image.UserImage.idなどにして呼び込む
    user_images = (
        db.session.query(User, UserImage)
        .join(UserImage)
        .filter(User.id == UserImage.user_id)
        .all()
    )
    # タグ一覧を取得する
    user_image_tag_dict = {}
    for user_image in user_images:
        # 画像に紐づくタグ一覧を取得
        user_image_tags = (
            db.session.query(UserImageTag)
            .filter(UserImageTag.user_image_id == user_image.UserImage.id)
            .all()
        )
        user_image_tag_dict[user_image.UserImage.id] = user_image_tags
        # 物体検知フォームをインスタンス化
        detector_form = DetectorForm()
    return render_template(
        "detector/index.html",
        user_images=user_images,
        # タグ一覧をテンプレートに渡す
        user_image_tag_dict=user_image_tag_dict,
        # 物体検知フォームをテンプレートに渡す
        detector_form=detector_form,
    )


@dt.route("/images/<path:filename>")
def image_file(filename):
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)


@dt.route("/upload", methods=["GET", "POST"])
@login_required
def upload_image():
    form = UploadImageForm()
    if form.validate_on_submit():
        # アップロードされた画像ファイルの取得
        file = form.image.data

        # ファイル名と拡張子を取得してファイル名をuuidに変換
        ext = Path(file.filename).suffix
        image_uuid_file_name = str(uuid.uuid4()) + ext

        # 画像を保存する
        image_path = Path(current_app.config["UPLOAD_FOLDER"], image_uuid_file_name)
        file.save(image_path)

        # DB保存
        user_image = UserImage(user_id=current_user.id, image_path=image_uuid_file_name)
        db.session.add(user_image)
        db.session.commit()
        return redirect(url_for("detector.index"))
    return render_template("detector/upload.html", form=form)


def make_color(labels):
    # 枠線の色をランダムに決定
    colors = [
        [random.randint(0, 255) for _ in range(3)] for _ in labels
    ]  # 0-255の整数を３つ生成、labelsの要素数と同じ個数生成される
    color = random.choice(colors)
    return color


def make_line(result_image):
    # 枠線を作成
    line = round(0.002 * max(result_image.shape[0:2])) + 1
    return line


def draw_lines(c1, c2, result_image, line, color):
    # 四角形の枠線を画像に追記
    cv2.rectangle(result_image, c1, c2, color, thickness=line)
    return cv2


def draw_texts(result_image, line, c1, cv2, color, labels, label):
    # 検知したテキストラベルを画像に追記
    display_txt = f"{labels[label]}"
    font = max(line - 1, 1)
    t_size = cv2.getTextSize(display_txt, 0, fontScale=line / 3, thickness=font)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(result_image, c1, c2, color, -1)
    cv2.putText(
        result_image,
        display_txt,
        (c1[0], c1[1] - 2),
        0,
        line / 3,
        [225, 255, 255],
        thickness=font,
        lineType=cv2.LINE_AA,
    )
    return cv2


def exec_detect(target_image_path):
    # ラベルの読み込み
    labels = current_app.config["LABELS"]
    #
    image = Image.open(target_image_path)
    #
    image_tensor = torchvision.transforms.functional.to_tensor(image)

    #
    model = torch.load(Path(current_app.root_path, "detector", "model.pt"))
    #
    model = model.eval()
    #
    output = model([image_tensor])[0]

    tags = []
    result_image = np.array(image.copy())
    #
    for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
        if score > 0.5 and labels[label] not in tags:
            #
            color = make_color(labels)
            #
            line = make_line(result_image)
            #
            c1 = (int(box[0]), int(box[1]))
            c2 = (int(box[2]), int(box[3]))
            #
            cv2 = draw_lines(c1, c2, result_image, line, color)
            cv2 = draw_texts(result_image, line, c1, cv2, color, labels, label)
            tags.append(labels[label])
    detected_image_file_name = str(uuid.uuid4()) + ".jpg"
    #
    detected_image_file_path = str(
        Path(current_app.config["UPLOAD_FOLDER"], detected_image_file_name)
    )

    #
    cv2.imwrite(detected_image_file_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    return tags, detected_image_file_name


def save_detected_image_tags(user_image, tags, detected_image_file_name):
    #
    user_image.image_path = detected_image_file_name
    #
    user_image.is_detected = True
    db.session.add(user_image)

    #
    for tag in tags:
        user_image_tag = UserImageTag(user_image_id=user_image.id, tag_name=tag)
        db.session.add(user_image_tag)
    db.session.commit()


@dt.route("/detect/<string:image_id>", methods=["POST"])
#
@login_required
def detect(image_id):
    #
    user_image = db.session.query(UserImage).filter(UserImage.id == image_id).first()
    if user_image is None:
        flash("物体検知対象の画像がありません")
        return redirect(url_for("detector.index"))
    #
    target_image_path = Path(current_app.config["UPLOAD_FOLDER"], user_image.image_path)

    #
    tags, detected_image_file_name = exec_detect(target_image_path)

    try:
        #
        save_detected_image_tags(user_image, tags, detected_image_file_name)
    except SQLAlchemyError as e:
        flash("物体検知処理でエラーが発生しました")
        #
        db.session.rollback()
        #
        current_app.logger.error(e)
        return redirect(url_for("detector.index"))
    return redirect(url_for("detector.index"))
