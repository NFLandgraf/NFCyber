from moviepy.editor import (VideoFileClip,ImageClip,TextClip,CompositeVideoClip)
from telegram.ext import ApplicationBuilder, MessageHandler, filters
from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageOps
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import zipfile
import logging
import json
import os


path = Path(r"C:\Users\landgrafn\NFCyber\Swellpass")
token = "8664194994:AAH4Iii3eyWkyIkE-7bAp0p1j7lvNv3aH5M"


def setup_message_logger(path):
    log_path = Path(path) / "Bot_log.log"
    logger = logging.getLogger("telegram_messages")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def create_ticket_v1(user_name, studio, studio_address, date, time):

    template = path / "Users" / user_name / "Template.jpg"
    user_pic = path / "Users" / user_name / "Pic.jpg"


    def load_font(size: int) -> ImageFont.FreeTypeFont:
        font_path = path / "Roboto" / "Roboto-Medium.ttf"
        try:
            if font_path and os.path.exists(font_path):
                return ImageFont.truetype(str(font_path), size)
            # Try a common system font
            return ImageFont.truetype("arial.ttf", size)
        except Exception as e:
            return ImageFont.load_default()

    background = "#F5F5F5"

    # Größe vom Screenshot übernehmen
    base = Image.open(template).convert("RGB")
    W, H = base.size

    # sizes
    leiste_y = int(0.1 * H)
    rectop_y = int(0.122 * H)
    bottom_y = int(0.63 * H)
    avatar_diameter = 0.4
    avatarring_diameter = 0.015

    # Leerer Hintergrund in sehr hellem Grau wie im Screenshot
    bg = Image.new("RGB", (W, H), background)
    draw = ImageDraw.Draw(bg)

    # Höhen der Zeilen
    rectangle_height = bottom_y - rectop_y
    line0 = int(0.05 * H) # Check-In Ticket
    line1 = int(rectop_y + rectangle_height * (42/100)) # Mitglied
    line2 = int(rectop_y + rectangle_height * (46/100))
    line3 = int(rectop_y + rectangle_height * (58/100)) # date
    line4 = int(rectop_y + rectangle_height * (62/100))
    line5 = int(rectop_y + rectangle_height * (71/100)) # Line
    line6 = int(rectop_y + rectangle_height * (77/100)) # Netzwerkpartner
    line7 = int(rectop_y + rectangle_height * (161/200))
    line8 = int(rectop_y + rectangle_height * (85/100))

    # Typographie
    small =         load_font(int(0.012 * H))
    normal =        load_font(int(0.014 * H))
    big =           load_font(int(0.016 * H))
    title_font =    load_font(int(0.020 * H))

    
    def backgrounds(W, H, leiste_y, rectop_y, bottom_y, avatar_diameter, avatarring_diameter, user_pic):
        # draws background, circles, rectangles, etc

        def draw_rounded_rect_gradient(img, box, radius, top_color, bottom_color):
            
            x0, y0, x1, y1 = box
            w, h = x1 - x0, y1 - y0

            # Ziel-Farben parsen
            c0 = ImageColor.getrgb(top_color)
            c1 = ImageColor.getrgb(bottom_color)

            # Diagonalen-Vektor (links-oben -> rechts-unten)
            dx, dy = max(w - 1, 1), max(h - 1, 1)
            denom = dx*dx + dy*dy

            # Verlauf auf eigenem Bild zeichnen
            grad = Image.new("RGB", (w, h))
            px = grad.load()
            for y in range(h):
                # Precompute y-Anteil für Geschwindigkeit
                yterm = y * dy
                for x in range(w):
                    # Projektion des Punkts (x,y) auf den Diagonalvektor -> t in [0..1]
                    t = (x * dx + yterm) / denom
                    # Linear interpolieren
                    r = int(c0[0] * (1 - t) + c1[0] * t)
                    g = int(c0[1] * (1 - t) + c1[1] * t)
                    b = int(c0[2] * (1 - t) + c1[2] * t)
                    px[x, y] = (r, g, b)

            # Runde Ecken als Maske
            mask = Image.new("L", (w, h), 0)
            mdraw = ImageDraw.Draw(mask)
            mdraw.rounded_rectangle((0, 0, w, h), radius=radius, fill=255)

            # Auf Zielbild einfügen
            img.paste(grad, (x0, y0), mask)


        avatar_d = int(avatar_diameter * W)             # inner photo diameter
        ring_w   = int(avatarring_diameter * W)         # width of the white ring
        ring_d   = avatar_d + 2*ring_w                  # outer diameter including ring
        avatar_x = (W - avatar_d) // 2
        avatar_y = rectop_y

        # App-Leiste oben (hellgrau)
        draw.rectangle((0, 0, W, leiste_y), fill="#E2E2E2")

        # Black arrow
        arrow_size = int(0.02 * W)                          # overall size of arrow
        arrow_x = int(0.06 * W)                             # distance from left
        arrow_width = int(0.007 * W)
        arrow_line = int(line0 + 0.01*H)

        # Coordinates
        p_mid = (arrow_x, arrow_line)                                # left tip
        p_top = (arrow_x + arrow_size, arrow_line - arrow_size)      # top tip
        p_bot = (arrow_x + arrow_size, arrow_line + arrow_size)      # bot tip
        p_end = (arrow_x + arrow_size*2.2, arrow_line)              # right tip

        # Draw two lines: top→mid and bot→mid
        draw.line([p_end, p_mid], fill="black", width=arrow_width)
        draw.line([p_top, p_mid], fill="black", width=arrow_width)
        draw.line([p_bot, p_mid], fill="black", width=arrow_width)

        # 1) Draw turquoise rectangle
        card_margin_x = int(0.05 * W)
        card_top = rectop_y + int(avatar_d/2)
        card_bottom = bottom_y
        card_box = (card_margin_x, card_top, W - card_margin_x, card_bottom)
        draw_rounded_rect_gradient(bg, card_box, radius=int(0.025 * W),top_color="#00A6B4", bottom_color="#006C79")

        # 2) Draw white circle (the border/ring)
        ring = Image.new("RGBA", (ring_d, ring_d), (0,0,0,0))
        ring_draw = ImageDraw.Draw(ring)
        ring_draw.ellipse((0,0,ring_d,ring_d), fill=(245,245,245,255))
        bg.paste(ring, (avatar_x - ring_w, avatar_y - ring_w), ring)

        # 3) Load and crop your photo
        photo = Image.open(user_pic).convert("RGB")
        photo = ImageOps.exif_transpose(photo).convert("RGB")
        min_side = min(photo.width, photo.height)
        left = (photo.width - min_side)//2
        top  = (photo.height - min_side)//2
        photo = photo.crop((left, top, left+min_side, top+min_side))
        photo = photo.resize((avatar_d, avatar_d), Image.LANCZOS)

        # 4) Create circular mask for photo
        mask = Image.new("L", (avatar_d, avatar_d), 0)
        ImageDraw.Draw(mask).ellipse((0,0,avatar_d,avatar_d), fill=255)
        bg.paste(photo, (avatar_x, avatar_y), mask)

        # 5) kleine Aussparung unten
        notch_r = int(0.035 * W)
        notch_center = ((card_box[0] + card_box[2]) // 2, card_box[3])
        notch_mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(notch_mask).ellipse((notch_center[0] - notch_r, notch_center[1] - notch_r, notch_center[0] + notch_r, notch_center[1] + notch_r), fill=255)
        notch_cut = Image.new("RGB", (W, H), background)
        bg.paste(notch_cut, (0, 0), notch_mask)

        # Linke und rechte Spalte
        left_x = card_box[0] + int(0.05 * W)
        right_x = card_box[2] - int(0.28 * W)

        return card_box, left_x, right_x
    
    def beschriftung(user_name, date, time, studio, studio_address):

        # Titel links in der Appbar
        draw.text((int(0.15 * W), line0), "Check-in Ticket", fill="#000000", font=title_font)

        # Beschriftungen
        draw.text((left_x, line1), "Mitglied", fill="white", font=small)
        draw.text((right_x, line1), "Status", fill="white", font=small)
        draw.text((left_x, line2), user_name, fill="white", font=big)
        draw.text((left_x, line3), "date", fill="white", font=small)
        draw.text((right_x, line3), "time", fill="white", font=small)
        draw.text((left_x, line4), date, fill="white", font=big)
        draw.text((right_x, line4), time, fill="white", font=big)

        # Netzwerkpartner
        draw.text((left_x, line6), "Netzwerkpartner", fill="white", font=small)
        draw.text((left_x, line7), studio, fill="white", font=big)
        draw.text((left_x, line8), studio_address, fill="white", font=normal)


        # Trennlinie (gepunktet)
        for x in range(card_box[0] + 10, card_box[2] - 10, 16):
            draw.line((x, line5, x + 8, line5), fill=(255, 255, 255, 200), width=3)

    def status_gif(bg):

        def draw_badge_frame(bg_base: Image.Image, alpha: int) -> Image.Image:
            # create the badge with transparency
            frame = bg_base.copy()

            # Create RGBA image for badge
            badge_img = Image.new("RGBA", (badge_w, badge_h), (0, 0, 0, 0))
            badge_draw = ImageDraw.Draw(badge_img)
            corner_round = int(badge_h / 5)  # make larger for less round
            badge_draw.rounded_rectangle((0, 0, badge_w, badge_h), radius=corner_round, fill=(74, 223, 131, alpha))

            # Measure text size
            bbox = badge_draw.textbbox((0, 0), "CHECKED-IN", font=normal)
            text_w, text_h = bbox[2]-bbox[0], bbox[3]-bbox[1]

            # Draw text with alpha
            text_x, text_y = (badge_w - text_w)//2, (badge_h - text_h)//2 - badge_h*0.1   # centered position in badge
            badge_draw.text((text_x, text_y), "CHECKED-IN", font=normal, fill=(11, 97, 94, alpha))
            frame.paste(badge_img, (badge_x, badge_y), badge_img)

            return frame

        badge_w, badge_h = int(0.21 * W), int(0.023 * H)
        badge_x, badge_y = right_x, line2

        # Build fade sequence: steps frames fade out, steps frames fade back in
        steps = 15
        alphas = list(reversed([int(200 * i/steps) for i in range(steps+1)]))  # 200→0
        alphas += list([int(200 * i/steps) for i in range(1, steps)])          # 0→200 again

        frames = [draw_badge_frame(bg, a) for a in alphas]
        durations = [40] * len(frames)   # blink speed, the higher the slower

        # Save as animated GIF
        gif_path = path / "Ticket.gif"
        frames[0].save(str(gif_path), save_all=True, append_images=frames[1:], format="GIF", duration=durations, loop=0, disposal=2)

        return gif_path

    card_box, left_x, right_x = backgrounds(W, H, leiste_y, rectop_y, bottom_y, avatar_diameter, avatarring_diameter, user_pic)
    beschriftung(user_name, date, time, studio, studio_address)
    gif_path = status_gif(bg)

    return gif_path

def create_ticket_v2(user_name, studio, studio_address, date, time):

    def crop_video(video_path, template_path):

        pic_center = (589, 1278)
        
        # we want to have a video that has the same dimensions as the screen template
        video = VideoFileClip(video_path)
        video_width, video_height = video.w, video.h
        with Image.open(template_path) as img:
            target_width, target_height = img.size
        
        if target_height > video_height:
            raise ValueError("Template height must be <= video height")

        # cuts half of the delta_height from the top and half from the bottom
        delta_h = video_height - target_height
        crop_top = delta_h // 2
        crop_bottom = delta_h - crop_top
        x1, x2 = 0, video_width
        y1, y2 = crop_top, video_height - crop_bottom
        video_crop = video.cropped(x1=x1, y1=y1, x2=x2, y2=y2)

        pic_center_x, pic_center_y = pic_center
        new_pic_center = (pic_center_x, pic_center_y - crop_top)

        return video_crop, new_pic_center

    def get_pic(path_pic):

        # crop pic and make a circle out of it, fitting perfectly into the hole
        img = Image.open(path_pic).convert("RGBA")
        img_w, img_h = img.size
        pic_dim = min(img_w, img_h)

        left = (img_w - pic_dim) // 2
        top = (img_h - pic_dim) // 2
        right = left + pic_dim
        bottom = top + pic_dim
        center_crop = img.crop((left, top, right, bottom))

        mask = Image.new("L", (pic_dim, pic_dim), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, pic_dim, pic_dim), fill=255)
        pic_round = Image.new("RGBA", (pic_dim, pic_dim), (0, 0, 0, 0))
        pic_round.paste(center_crop, (0, 0), mask)

        return pic_round, pic_dim

    def overlay_pic(video_crop, pic_round, new_pic_center, pic_dim):
        x, y = new_pic_center
        x_pos = int(x - pic_dim / 2)
        y_pos = int(y - pic_dim / 2)

        overlay = ImageClip(np.array(pic_round), transparent=True).with_duration(video_crop.duration).with_position((x_pos, y_pos))
        return overlay

    def add_texts(video_crop, user_name, studio, studio_address, date, time):

        def clip_text(text, font_path, font_size, box_width, stroke_width=0):
            
            # if text is longer than box, makes the right part that is too long '...'
            clip = TextClip(text=text, font=font_path, font_size=font_size, stroke_width=stroke_width, method="label")
            if clip.w <= box_width:
                return text

            left, right = 0, len(text)
            best = "..."
            while left <= right:
                mid = (left + right) // 2
                candidate = text[:mid].rstrip() + "..."
                clip = TextClip(text=candidate, font=font_path, font_size=font_size, stroke_width=stroke_width, method="label")
                if clip.w <= box_width:
                    best = candidate
                    left = mid + 1
                else:
                    right = mid - 1
            return best

        # change parameters accordingly
        box_width_videowidth = 0.85
        box_height = 100
        fontsize_user = 75
        fontsize_info = 42
        gray_dark_255 = 50
        gray_ligh_255 = 140
        strokewidth_user = 0
        strokewidth_info = 0.1
        y_posi_user = 0.67
        y_posi_studio = 0.71
        y_posi_datetime = 0.74

        font_medium = str(path / "Roboto" / "Roboto-Medium.ttf")
        video_width, video_height = video_crop.w, video_crop.h
        box_width = int(video_width * box_width_videowidth)
        duration = video_crop.duration
        text_clips = []
        gray_light = (gray_ligh_255, gray_ligh_255, gray_ligh_255)
        gray_dark = (gray_dark_255, gray_dark_255, gray_dark_255)

        # user_name
        clip = TextClip(
            text=user_name,
            font=font_medium,
            font_size=fontsize_user,
            color=gray_dark,
            stroke_width=strokewidth_user,
            size=(box_width, box_height),
            method="caption").with_duration(duration)
        x = video_width / 2 - box_width / 2
        y = video_height * y_posi_user - box_height / 2
        text_clips.append(clip.with_position((x, y)))

        # studio
        clip = TextClip(
            text=clip_text(studio, font_medium, fontsize_info, box_width, stroke_width=strokewidth_info),
            font=font_medium,
            font_size=fontsize_info,
            color=gray_light,
            stroke_width=strokewidth_info,
            size=(box_width, box_height),
            method="caption",
            text_align="left").with_duration(duration)
        x = video_width / 2 - box_width / 2
        y = video_height * y_posi_studio - box_height / 2
        text_clips.append(clip.with_position((x, y)))

        # date + time
        clip = TextClip(
            text=f'{date}, {time}',
            font=font_medium,
            font_size=fontsize_info,
            color=gray_light,
            stroke_width=strokewidth_info,
            size=(box_width, box_height)).with_duration(duration)
        x = video_width / 2 - box_width/2
        y = video_height * y_posi_datetime - box_height / 2
        text_clips.append(clip.with_position((x, y)))

        return text_clips

    # paths to use
    path_anim = path / 'Templates' / 'ticket_template.mp4'
    path_pic = path / 'Users' / user_name / 'Pic.jpg'
    path_template = path / 'Users' / user_name / 'Template.jpg'
    path_gif = path / 'Ticket.gif'

    # executives
    video_crop, new_pic_center = crop_video(path_anim, path_template)
    pic_round, pic_dim = get_pic(path_pic)
    video_overlay_pic = overlay_pic(video_crop, pic_round, new_pic_center, pic_dim)
    video_overlay_txt = add_texts(video_crop, user_name, studio, studio_address, date, time)
    
    final = CompositeVideoClip([video_crop, video_overlay_pic] + video_overlay_txt, size=video_crop.size)
    final = final.with_duration(video_crop.duration).with_fps(30)
    final.write_gif(str(path_gif), fps=30, logger=None)

    return path_gif

def reset_state(state):
    state['step'] = 'login'
    state['user_login'] = None
    state['user_name'] = None
    state['studio'] = None
    state['studio_address'] = None
    state['date'] = None
    state['time'] = None

def parse_date(msg):
    now = datetime.now()
    if msg == 'today':
        return now.date()
    if msg == 'tomorrow':
        return (now + timedelta(days=1)).date()
    return None

def parse_time(msg):
    now = datetime.now()
    if msg == 'now':
        return now.strftime("%H:%M")
    if msg.startswith('in '):
        try:
            minutes = int(msg.split()[1])
            return (now + timedelta(minutes=minutes)).strftime("%H:%M")
        except ValueError:
            return None
    if msg.startswith('at '):
        try:
            time_str = msg.split()[1]
            t = datetime.strptime(time_str, "%H:%M")
            return now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0).strftime("%H:%M")
        except ValueError:
            return None
    return None

def zip_gif(gif_path):
    gif_path = Path(gif_path)
    zip_path = gif_path.with_suffix('.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(gif_path, arcname=gif_path.name)
    return zip_path

def studio_list_text(state):
    a = build_summary(state)
    b = '\n\nWhere?\n' + '\n'.join(f'{i}: {s["name"]}' for i, s in enumerate(all_studios))
    return a+b

def date_prompt(state):
    a = build_summary(state)
    b = '\n\nWhen?\n"today" or "tomorrow"'
    return a+b

def time_prompt(state):
    a = build_summary(state)
    b = '\n\nWhen?\n"now", "in -min-" or "at -hh:mm-"'
    return a+b

def restart_timeout(update, context):
    timeout_sec = 300

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    for job in context.job_queue.get_jobs_by_name(f"timeout_{user_id}"):
        job.schedule_removal()

    context.job_queue.run_once(timeout_callback, when=timeout_sec, name=f"timeout_{user_id}", chat_id=chat_id, user_id=user_id)

def cancel_timeout(context, user_id):
    for job in context.job_queue.get_jobs_by_name(f"timeout_{user_id}"):
        job.schedule_removal()


def build_summary(state):
    if state['studio'] is None:
        return (f"<b>Who:</b>  {state['user_name']}\n"
                f"<b>Gym:</b>  {state['studio']}\n"
                f"<b>Date:</b>  {state['date']}\n"
                f"<b>Time:</b> {state['time']}")
    else:
        return (f"<b>Who:</b>  {state['user_name']}\n"
                f"<b>Gym:</b>  {state['studio']}, {state['studio_address']}\n"
                f"<b>Date:</b>  {state['date']}\n"
                f"<b>Time:</b> {state['time']}")

def build_summary_confirm(state):
    return (f"<b>0_Who:</b>  {state['user_name']}\n"
            f"<b>1_Gym:</b>  {state['studio']}, {state['studio_address']}\n"
            f"<b>2_Date:</b>  {state['date']}\n"
            f"<b>3_Time:</b> {state['time']}")
    
async def timeout_callback(context):
    reset_state(context.user_data)
    await context.bot.send_message(chat_id=context.job.chat_id,text="reset")

async def handle_login(update, state, msg):
    user = next((u for u in all_users if u["login"] == msg), None)
    if user is not None:
        state['user_login'] = user["login"]
        state['user_name'] = user["name"]
        state['step'] = 'choose_studio'

        await update.message.reply_text(studio_list_text(state), parse_mode="HTML")
    else:
        await update.message.reply_text('...')

async def handle_choose_studio(update, state, msg):
    if msg.isdigit() and int(msg) < len(all_studios):
        idx = int(msg)
        state['studio'] = all_studios[idx]['name']
        state['studio_address'] = all_studios[idx]['address']
        state['step'] = 'choose_date'

        await update.message.reply_text(date_prompt(state), parse_mode="HTML")
    else:
        await update.message.reply_text(studio_list_text(state), parse_mode="HTML")

async def handle_choose_date(update, state, msg):
    parsed_date = parse_date(msg)
    if parsed_date is not None:
        state['date'] = parsed_date.strftime("%d.%m.%Y")
        state['step'] = 'choose_time'

        await update.message.reply_text(time_prompt(state), parse_mode="HTML")
    else:
        await update.message.reply_text(date_prompt(state), parse_mode="HTML")

async def handle_choose_time(update, state, msg):
    parsed_time = parse_time(msg)
    if parsed_time is not None:
        state['time'] = parsed_time
        state['step'] = 'confirm'

        await update.message.reply_text(build_summary_confirm(state) + '\n\nConfirm? Version?\n"new", "old" or change sth with "-number-"', parse_mode="HTML")
    else:
        await update.message.reply_text(time_prompt(state), parse_mode="HTML")

async def confirm(update, state, msg):
    if msg == 'old':
        await send_ticket(update, state, 'v1')
    elif msg == 'new':
        await send_ticket(update, state, 'v2')
    
    elif msg == '0':
        reset_state(state)
        await update.message.reply_text('Reset')
        return
    elif msg == '1':
        state['step'] = 'choose_studio'
        state['studio'] = None
        state['studio_address'] = None
        state['date'] = None
        state['time'] = None
        await update.message.reply_text(studio_list_text(state), parse_mode="HTML")
    elif msg == '2':
        state['step'] = 'choose_date'
        state['date'] = None
        state['time'] = None
        await update.message.reply_text(date_prompt(state), parse_mode="HTML")
    elif msg == '3':
        state['step'] = 'choose_time'
        state['time'] = None
        await update.message.reply_text(time_prompt(state), parse_mode="HTML")
    else:
        await update.message.reply_text(build_summary_confirm(state) + '\n\nConfirm?\n"y" or change "<nb>"', parse_mode="HTML")

async def send_ticket(update, state, version):

    await update.message.reply_text(build_summary(state) + '\n\ncreating ticket...(13s)', parse_mode="HTML")
        
    # create and send ticket
    if version == 'v1':
        gif_path = create_ticket_v1(state['user_name'], state['studio'], state['studio_address'], state['date'], state['time'])
    elif version == 'v2':
        gif_path = create_ticket_v2(state['user_name'], state['studio'], state['studio_address'], state['date'], state['time'])
    
    zip_path = zip_gif(gif_path)
    with zip_path.open('rb') as f:
        await update.message.reply_document(document=f, filename=zip_path.name)
    await update.message.reply_text(build_summary_confirm(state) + '\n\nchange sth type the -nb-\nGood training!', parse_mode="HTML")


async def antwort(update, context):

    # check if its actually a real message
    if not update.message or not update.message.text:
        return

    # cleans the message
    msg = update.message.text.lower().strip()
    user_id = update.effective_user.id if update.effective_user else "unknown"
    username = update.effective_user.username if update.effective_user and update.effective_user.username else "unknown"
    chat_id = update.effective_chat.id if update.effective_chat else "unknown"

    # logging
    msg_logger.info(f"user_id={user_id} | username={username} | chat_id={chat_id} | text={msg!r}")

    # this is the dictionary why the bot remebers the conversation, data is stored in here
    state = context.user_data

    # reset everything
    if msg == 'reset':
        reset_state(state)
        await update.message.reply_text('Reset')
        cancel_timeout(context, update.effective_user.id)
        return

    # if a new run is initiated or the user resets it
    if 'step' not in state:
        reset_state(state)
    
    # depending on the current step, it runs a different function
    handlers = {
        'login': handle_login,
        'choose_studio': handle_choose_studio,
        'choose_date': handle_choose_date,
        'choose_time': handle_choose_time,
        'confirm': confirm}
    handler = handlers.get(state['step'])
    await handler(update, state, msg)

    # restart the timer for timeout
    if state['user_login'] is not None:
        restart_timeout(update, context)

msg_logger = setup_message_logger(path)
all_studios = json.load(open(path / "Users" / "data_studios.json", encoding="utf-8"))
all_users = json.load(open(path / "Users" / "data_users.json", encoding="utf-8"))

app = ApplicationBuilder().token(token).build()
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, antwort))
app.run_polling()
