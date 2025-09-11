#%%
# Swellpass
import imaplib, email, smtplib, time, os
from email.message import EmailMessage
from email.header import decode_header, make_header
from email.utils import parseaddr, formataddr, parsedate_to_datetime
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageOps
import mimetypes
from pathlib import Path
from datetime import datetime


path = "C:\\Users\\landgrafn\\NFCyber\\Swellpass"
load_dotenv()
IMAP_HOST = os.getenv("IMAP_HOST")
IMAP_PORT = int(os.getenv("IMAP_PORT", "993"))
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
SENDER_NAME = os.getenv("SENDER_NAME", "Swellpass Bot")


def render_ticket(template, user_pic, user_name, datum, zeit, partner_name, partner_address) -> Image.Image:

    def load_font(size: int) -> ImageFont.FreeTypeFont:
        font_path = path + "\\Roboto\\Roboto-Medium.ttf"
        try:
            if font_path and os.path.exists(font_path):
                return ImageFont.truetype(font_path, size)
            # Try a common system font
            return ImageFont.truetype("arial.ttf", size)
        except Exception as e:
            print("Falling back to default font:", e)
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
    line3 = int(rectop_y + rectangle_height * (58/100)) # Datum
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
    
    def beschriftung(user_name, datum, zeit, partner_name, partner_address):

        # Titel links in der Appbar
        draw.text((int(0.15 * W), line0), "Check-in Ticket", fill="#000000", font=title_font)

        # Beschriftungen
        draw.text((left_x, line1), "Mitglied", fill="white", font=small)
        draw.text((right_x, line1), "Status", fill="white", font=small)
        draw.text((left_x, line2), user_name, fill="white", font=big)
        draw.text((left_x, line3), "Datum", fill="white", font=small)
        draw.text((right_x, line3), "Zeit", fill="white", font=small)
        draw.text((left_x, line4), datum, fill="white", font=big)
        draw.text((right_x, line4), zeit, fill="white", font=big)

        # Netzwerkpartner
        draw.text((left_x, line6), "Netzwerkpartner", fill="white", font=small)
        draw.text((left_x, line7), partner_name, fill="white", font=big)
        draw.text((left_x, line8), partner_address, fill="white", font=normal)


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
        gif_path = path + "\\Ticket.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], format="GIF", duration=durations, loop=0, disposal=2)
        print("GIF saved")

        return gif_path

    card_box, left_x, right_x = backgrounds(W, H, leiste_y, rectop_y, bottom_y, avatar_diameter, avatarring_diameter, user_pic)
    beschriftung(user_name, datum, zeit, partner_name, partner_address)
    gif_path = status_gif(bg)

    return gif_path

def ident_user(user):

    if user == 'Nico':
        user_name = 'Nicolas Landgraf'
    elif user == 'Jonny':
        user_name = 'Jonathan Jotter'
    elif user == 'Lorenz':
        user_name = 'Lorenz Baier'
    elif user == 'Stanni':
        user_name = 'Constanze Gathen'
    elif user == 'Nana':
        user_name = 'Johanna Gentz'
    else:
        print('No user identified')
    
    print(user_name)
    template = path + f"\\Users\\{user_name}\\Template.jpg"
    user_pic = path + f"\\Users\\{user_name}\\Pic.jpg"

    return template, user_pic, user_name

def reply_with_file(to_address: str, file_path: str):

    # check gif path
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Attachment not found: {file_path}")

    ctype, encoding = mimetypes.guess_type(str(path))
    if ctype is None:
        maintype, subtype = "application", "octet-stream"
    else:
        maintype, subtype = ctype.split("/", 1)

    msg = EmailMessage()
    msg["Subject"] = f"Re: {str(make_header(decode_header('Swellpass Ticket')))}"
    msg["From"] = formataddr((SENDER_NAME, EMAIL_USER))
    msg["To"] = to_address
    msg.set_content("Swello")

    with open(path, "rb") as f:
        msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=path.name)

    with smtplib.SMTP(SMTP_HOST, int(SMTP_PORT)) as s:
        s.ehlo()
        s.starttls()
        s.ehlo()
        s.login(EMAIL_USER, EMAIL_PASS)
        s.send_message(msg)
    print('Ticket sent\n')

def process_message(imap, mail_id):

    status, data = imap.fetch(mail_id, "(RFC822)")
    if status != "OK":
        return
    msg_in = email.message_from_bytes(data[0][1])
    from_addr = parseaddr(msg_in.get("From"))[1]

    # Date
    date_raw = msg_in.get("Date", "")
    try:
        msg_datetime = parsedate_to_datetime(date_raw)
        print(msg_datetime.strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as e:
        print("Could not parse date:", date_raw, e)
    print(from_addr)
    
    # Extract plain text
    payload_text = ""
    if msg_in.is_multipart():
        for part in msg_in.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition") or "")
            if ctype == "text/plain" and "attachment" not in disp:
                payload_text = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="ignore")
                break
    else:
        payload_text = msg_in.get_payload(decode=True).decode(msg_in.get_content_charset() or "utf-8", errors="ignore")

    # get info from mail
    lines = [line.strip() for line in payload_text.splitlines() if line.strip()]
    partner_name = lines[0] if len(lines) > 0 else ""
    partner_address = lines[1] if len(lines) > 1 else ""
    now = datetime.now()
    datum = lines[2] if len(lines) > 2 else str(now.date().strftime("%d.%m.%y"))
    zeit = lines[3] if len(lines) > 3 else str(now.time().strftime("%H:%M"))

    # identify user
    user = msg_in.get("Subject", "")
    template, user_pic, user_name = ident_user(user)
    print(partner_name)
    print(partner_address)
    gif_path = render_ticket(template, user_pic, user_name, datum, zeit, partner_name, partner_address)

    reply_with_file(from_addr, gif_path)

def watch_loop():
    print('Swellbot ready\n\n')
    while True:
        try:
            with imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT) as imap:
                imap.login(EMAIL_USER, EMAIL_PASS)
                imap.select("INBOX")

                # nur ungelesene, die an UNS gingen
                status, data = imap.search(None, '(UNSEEN)')
                if status == "OK":
                    ids = data[0].split()
                    for mid in ids:
                        process_message(imap, mid)
                        imap.store(mid, '+FLAGS', '\\Seen')
                imap.logout()

        except Exception as e:
            print("Fehler im Watcher:", e)
        time.sleep(5)   # in seconds

watch_loop()
