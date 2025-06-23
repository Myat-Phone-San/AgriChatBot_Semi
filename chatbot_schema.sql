-- Database: agrichatbot
-- Generation Time: 2025-06-20
-- Character set: utf8mb4 COLLATE utf8mb4_unicode_ci

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `agrichatbot`
--

-- --------------------------------------------------------

--
-- Drop existing tables in reverse dependency order to avoid foreign key issues
--
DROP TABLE IF EXISTS `dash_cards`;
DROP TABLE IF EXISTS `options`; -- Drop options before answers if it links to answers
DROP TABLE IF EXISTS `answers`;
DROP TABLE IF EXISTS `sub_categories`;
DROP TABLE IF EXISTS `child_categories`;
DROP TABLE IF EXISTS `parent_categories`;
DROP TABLE IF EXISTS `unanswered_queries`;

-- --------------------------------------------------------

--
-- Table structure for table `parent_categories`
-- Broad categories like 'စပါး', 'ကော်ဖီ'
--

CREATE TABLE `parent_categories` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `name` TEXT COLLATE utf8mb4_unicode_ci NOT NULL,
  `created_at` TIMESTAMP NULL DEFAULT NULL,
  `updated_at` TIMESTAMP NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  FULLTEXT KEY `name_fulltext` (`name`(255))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------

--
-- Table structure for table `child_categories`
-- Specific topics under parent categories, like 'မိတ်ဆက်', 'စိုက်စနစ်', 'ပိုး', 'ရောဂါ', 'ပေါင်းပင်'
--

CREATE TABLE `child_categories` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `name` TEXT COLLATE utf8mb4_unicode_ci NOT NULL,
  `parent_category_id` BIGINT UNSIGNED DEFAULT NULL,
  `created_at` TIMESTAMP NULL DEFAULT NULL,
  `updated_at` TIMESTAMP NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  FOREIGN KEY (`parent_category_id`) REFERENCES `parent_categories` (`id`) ON DELETE CASCADE,
  FULLTEXT KEY `name_fulltext` (`name`(255))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------

--
-- Table structure for table `sub_categories`
-- Sub-topics under child categories, like 'ရွက်စားယင်' (under 'ပိုး'), or specific cultivation systems.
--

CREATE TABLE `sub_categories` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `name` TEXT COLLATE utf8mb4_unicode_ci NOT NULL,
  `child_category_id` BIGINT UNSIGNED DEFAULT NULL, -- This MUST reference an ID from `child_categories`
  `created_at` TIMESTAMP NULL DEFAULT NULL,
  `updated_at` TIMESTAMP NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  FOREIGN KEY (`child_category_id`) REFERENCES `child_categories` (`id`) ON DELETE CASCADE,
  FULLTEXT KEY `name_fulltext` (`name`(255))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------

--
-- Table structure for table `answers`
-- This table is now central. 'keyword' acts as the query text, 'name' is the answer.
-- It links directly to categories.
--

CREATE TABLE `answers` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `keyword` TEXT COLLATE utf8mb4_unicode_ci DEFAULT NULL, -- This acts as the "question text" / "query phrase" for training/matching
  `name` LONGTEXT COLLATE utf8mb4_unicode_ci NOT NULL, -- The actual answer content (can contain HTML, cleaned by Python app)
  `parent_category_id` BIGINT UNSIGNED DEFAULT NULL,
  `child_category_id` BIGINT UNSIGNED DEFAULT NULL,
  `sub_category_id` BIGINT UNSIGNED DEFAULT NULL,
  `image_path` VARCHAR(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL, -- Optional path to an image related to the answer
  `created_at` TIMESTAMP NULL DEFAULT NULL,
  `updated_at` TIMESTAMP NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `answers_parent_category_id_foreign` (`parent_category_id`),
  KEY `answers_child_category_id_foreign` (`child_category_id`),
  KEY `answers_sub_category_id_foreign` (`sub_category_id`),
  FOREIGN KEY (`parent_category_id`) REFERENCES `parent_categories` (`id`) ON DELETE SET NULL,
  FOREIGN KEY (`child_category_id`) REFERENCES `child_categories` (`id`) ON DELETE SET NULL,
  FOREIGN KEY (`sub_category_id`) REFERENCES `sub_categories` (`id`) ON DELETE SET NULL,
  FULLTEXT KEY `keyword_fulltext` (`keyword`(255)), -- Index 'keyword' for searching
  FULLTEXT KEY `name_fulltext` (`name`(255)) -- Index 'name' (answer content) for searching
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------

--
-- Table structure for table `options`
-- This table provides a way to link an answer to further "options" (follow-up questions/choices)
-- which in turn link to other answers.
--
CREATE TABLE `options` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `name` LONGTEXT COLLATE utf8mb4_unicode_ci NOT NULL, -- The text for the option (e.g., "စပါးမိတ်ဆက်")
  `answer_id` BIGINT UNSIGNED NOT NULL, -- The ID of the answer that presents this option
  `linkedAnswer_id` BIGINT UNSIGNED NOT NULL, -- The ID of the answer that this option leads to
  `created_at` TIMESTAMP NULL DEFAULT NULL,
  `updated_at` TIMESTAMP NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `options_answer_id_foreign` (`answer_id`),
  KEY `options_linkedanswer_id_foreign` (`linkedAnswer_id`),
  FOREIGN KEY (`answer_id`) REFERENCES `answers` (`id`) ON DELETE CASCADE,
  FOREIGN KEY (`linkedAnswer_id`) REFERENCES `answers` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------

--
-- Table structure for table `unanswered_queries`
-- Logs user queries that the chatbot could not answer from its database.
--

CREATE TABLE `unanswered_queries` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `query_text` TEXT COLLATE utf8mb4_unicode_ci NOT NULL,
  `created_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------

--
-- Table structure for table `dash_cards`
-- Stores information for dashboard cards/resource links in the UI.
--

CREATE TABLE `dash_cards` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `card_link` VARCHAR(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `image_path` VARCHAR(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `card_label` VARCHAR(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `created_at` TIMESTAMP NULL DEFAULT NULL,
  `updated_at` TIMESTAMP NULL DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------

--
-- Dumping data for tables
--
-- Note: 'keyword' in 'answers' table is now used as the primary query/topic for that answer.
-- The category IDs in 'answers' provide additional context.

-- Parent Categories
INSERT INTO `parent_categories` (`id`, `name`, `created_at`, `updated_at`) VALUES
(1, 'စပါး', '2024-08-12 00:45:16', '2024-08-12 00:45:16'),
(2, 'ကော်ဖီ', '2024-08-25 21:36:15', '2024-08-25 21:36:15'),
(3, 'ကြံ', '2024-08-26 00:02:11', '2024-08-26 00:02:11'),
(4, 'ရော်ဘာ', '2024-08-26 20:34:13', '2024-08-26 20:34:13'),
(5, 'ဝါ', '2024-08-27 00:07:22', '2024-08-27 00:07:22'),
(6, 'မြေပဲ', '2024-08-27 00:07:22', '2024-08-27 00:07:22');

-- Child Categories
INSERT INTO `child_categories` (`id`, `name`, `parent_category_id`, `created_at`, `updated_at`) VALUES
(6, 'မိတ်ဆက်', NULL, '2024-08-13 20:08:29', '2024-08-13 20:08:29'), -- General 'မိတ်ဆက်' (e.g., as a concept)
(7, 'စပါးမိတ်ဆက်', 1, '2024-08-13 20:13:06', '2024-08-13 20:13:06'), -- Specific 'စပါးမိတ်ဆက်' under 'စပါး'
(8, 'တိုက်ရိုက်မျိုးစေ့ချ(အခြောက်)', 1, '2024-08-13 20:48:56', '2024-08-13 20:48:56'), -- 'စပါး' related
(9, 'ပင်အမြင့်ဖောက်', 2, '2024-08-26 01:23:22', '2024-08-26 01:23:22'), -- 'ကော်ဖီ' related
(10, 'ပင်စည်ထိုး', 3, '2024-08-26 01:25:42', '2024-08-26 01:25:42'), -- 'ကြံ' related
(11, 'အဟာရနှင့်ရေလိုအပ်ချက်များ', 5, '2024-08-26 01:40:44', '2024-08-26 01:47:01'), -- 'ဝါ' related
(100, 'စိုက်စနစ်', NULL, '2024-08-13 20:33:58', '2024-08-13 20:33:58'), -- Moved from sub_categories, new ID
(101, 'ပိုး', NULL, '2024-08-18 19:56:55', '2024-08-18 19:56:55'), -- Moved from sub_categories, new ID
(102, 'ရောဂါ', NULL, '2024-08-25 20:04:20', '2024-08-25 20:04:20'), -- Moved from sub_categories, new ID
(103, 'ပေါင်းပင်', NULL, '2024-08-25 20:52:53', '2024-08-25 20:52:53'); -- Moved from sub_categories, new ID

-- Sub Categories (now correctly links to child_categories)
INSERT INTO `sub_categories` (`id`, `name`, `child_category_id`, `created_at`, `updated_at`) VALUES
(38, 'ရွက်စားယင်', 101, '2024-08-24 21:59:46', '2024-08-24 21:59:46'), -- Under 'ပိုး' (ID 101)
(40, 'အခြားဖျက်ပိုး', 101, '2024-08-25 20:38:58', '2024-08-25 20:38:58'), -- Under 'ပိုး' (ID 101)
(41, 'မြက်မုံညင်းအဝါ', 103, '2024-08-25 20:52:53', '2024-08-25 20:52:53'), -- Under 'ပေါင်းပင်' (ID 103)
(42, 'ဆစ်ပိုး', 101, '2024-08-25 20:57:42', '2024-08-25 20:57:42'); -- Under 'ပိုး' (ID 101)


-- Answers (keyword column now holds the query phrase)
INSERT INTO `answers` (`id`, `keyword`, `name`, `parent_category_id`, `child_category_id`, `sub_category_id`, `image_path`, `created_at`, `updated_at`) VALUES
(1, 'စပါး', '<p>စပါးနဲ့ပတ်သက်လို့ဘာများသိချင်ပါသလဲ</p>\n<p>စပါးနှင့်ပတ်သက်လို့ဘာများသိချင်ပါသလဲ</p>\n<p>၁ စပါးမိတ်ဆက်</p>\n<p>၂&nbsp; စပါးစိုက်စနစ်&nbsp;</p>\n<p>၃ စပါးမျိုးကွဲများ</p>\n<p>၄ စပါးတွင်ကျရောက်တတ်သောအင်းဆက်ပိုးများ</p>\n<p>၅&nbsp; စပါးတွင်ကျရောက်တက်သောရောဂါများ</p>\n<p>၆ စပါးတွင်ကျရောက်တတ်သောပေါင်းပင်များ</p>\n<p>၇ စပါးတွင်ကျရောက်တတ်သောအခြားဖျက်ပိုးများ</p>\n<p>၈<strong>&nbsp;</strong>စပါးအတွက် စိုက်ပျိုးရေးဆိုင်ရာ အလေ့အကျင့်ကောင်းများ (RiceGAP)</p>', 1, NULL, NULL, NULL, '2024-08-12 21:43:08', '2024-08-13 20:08:08'),
(2, 'စပါးမိတ်ဆက်', '<p>စပါးသီးနှံစိုက်ပျိုးမှုတွင် မိုးစပါး၊ နွေစပါးစိုက်ပျိုး ထုတ်လုပ် မှု များဆောင်ရွက်လျက်ရှိပြီး နှစ်စဉ်မိုးစပါး(၁၅)သန်းခန့်နှင့် နွေစပါး(၂.၅)သန်းကျော် စုစုပေါင်း (၁၇)သန်းကျော် စိုက်ပျိုးထုတ်လုပ်လျက်ရှိပါသည်။ လက်ရှိအနေဖြင့် ၂၀၂၄-၂၀၂၅ခုနှစ် အနေ ဖြင့် မိုးစပါးတွင် စိုက်ဧက(၁၅၀၀၅၈၁၈)လျာထားဆောင်ရွက်လျက်ရှိပြီး &nbsp;ယနေ့ထိ စိုက်ဧက (၇၅၃၈၉၃၆)စိုက်ပျိုးပြီးစီး၍ (၅၀.၂၄)ရာခိုင်နှုန်းဆောင်ရွက်ပြီးဖြစ်ပါသည်။ ယခင်နှစ် စိုက်ပျိုး ထွက်ရှိမှုအနေဖြင့် ၂၀၂၃-၂၀၂၄ မိုးစပါးတွင် စိုက်ဧက(၁၄၉၉၇၄၇၅)၊ ရိတ်ဧက (၁၄၉၆၅၄၇၆)၊ အထွက်နှုန်း(၇၈.၃၃)တင်းဖြင့် အထွက်တင်းပေါင်း (၁၁၇၂၂၆၂၉၉၂) ရရှိအောင် ဆောင်ရွက် နိုင်ခဲ့ပြီး ၂၀၂၃-၂၀၂၄ နွေစပါးအနေဖြင့် စိုက်ဧက(၂၇၄၅၅၃၈)၊ ရိတ်ဧက (၂၅၉၈၃၀၄)၊ အထွက် နှုန်း(၁၀၅.၀၁)တင်းဖြင့် အထွက်တင်းပေါင်း (၂၇၂၈၄၆၀၉၄) ရရှိအောင် ဆောင်ရွက်ထားရှိပြီး ဆက်လက်ရိတ်သိမ်းဆောင်ရွက်လျက်ရှိပါသည်။</p>', 1, 7, NULL, NULL, '2024-08-13 20:13:06', '2024-08-13 20:13:06'),
(3, 'မိတ်ဆက်', '<p>မိတ်ဆက်နှင့်ပတ်သက်ပြီး ဘယ်သီးနှံကိုသိချင်ပါသလဲ။</p>\n<p>(၁) စပါးမိတ်ဆက်</p>\n<p>(၂) ကော်ဖီမိတ်ဆက်</p>\n<p>(၃)ကြံမိတ်ဆက်</p>\n<p>(၄)ရော်ဘာမိတ်ဆက်</p>\n<p>(၅)ဝါမိတ်ဆက်</p>\n<p>(၆)မြေပဲမိတ်ဆက်</p>', NULL, 6, NULL, NULL, '2024-08-13 20:33:14', '2024-09-04 23:26:36'),
(4, 'စပါးစိုက်စနစ်', '<p>စပါးစိုက်စနစ်များမှာ</p>\n<p>စပါးတိုက်ရိုက်မျိုးစေ့ချ(အခြောက်)စိုက်စနစ်<br>စပါးတိုက်ရိုက်မျိုးစေ့ချ(အစို)စိုက်စနစ်</p>', 1, 100, NULL, NULL, '2024-08-13 20:41:24', '2024-08-14 20:28:59'), -- Linked to new child_category_id for 'စိုက်စနစ်'
(5, 'စပါးအခြောက်တိုက်ရိုက်မျိုးစေ့ချ', 'စပါးအခြောက်တိုက်ရိုက်မျိုးစေ့ချစိုက်ပျိုးနည်း (Dry Direct Seeding)\nစပါးအခြောက်တိုက်ရိုက်မျိုးစေ့ချစိုက်နည်းစနစ်အား မြန်မာနိုင်ငံတွင် ရေကြီးကွင်း၊ ရေနက်ကွင်းများ၌ စိုက်ပျိုးလေ့ရှိပါသည်။ ဤစနစ်သည် ရေအရင်းအမြစ်နည်းပါးသောနေရာများတွင် ထိရောက်မှုရှိပြီး ရိတ်သိမ်းမှုကို ပိုမိုမြန်ဆန်စေပါသည်။', 1, 8, NULL, NULL, '2025-06-20 00:42:00', '2025-06-20 00:42:00'),
(6, 'မြက်မုံညင်းအဝါ', 'အမည်: မြက်မုံညင်းအဝါ (Rice flatsedge) (Cyperus iria)\n\n**လက္ခဏာရပ်များ**:\n* ၎င်းသည် တစ်နှစ်ခံ ပေါင်းပင်တစ်မျိုးဖြစ်သည်။\n* အစေ့အားဖြင့် မျိုးပြားသည်။\n* မျိုးစေ့များသည် အစေ့ငုပ်နေနိုင်သော်လည်း ကြွေကျပြီး ၇၅ ရက်ကြာ၌ အပင်ပေါက်နိုင်သည်။\n* ပင်ငယ်၏ ထိပ်ဖျားသည် လှံသွားသကဲ့သို့ ချွန်နေသည်။\n* ပန်းပွင့်များသည် သေးငယ်ပြီး အဝါရောင်နှင့် အညိုရောင်ဖြစ်သည်။\n\n**ကျရောက်မှု**:\n* စိုထိုင်းသောမြေစေးနှင့် သဲမြေတွင် ကောင်းစွာရှင်သန်နိုင်သည်။\n* စပါးစိုက်ခင်းများတွင် အတွေ့ရများသော ပေါင်းပင်ဖြစ်သည်။', 1, 103, 41, NULL, '2025-06-20 00:42:00', '2025-06-20 00:42:00'), -- Linked to 'ပေါင်းပင်' child category (ID 103) and its own sub_category_id (41)
(7, 'ဝါမိတ်ဆက်', 'ဝါသီးနှံမိတ်ဆက်: ဝါသီးနှံသည် မြန်မာနိုင်ငံတွင် အဓိကစိုက်ပျိုးသီးနှံများထဲမှ တစ်ခုဖြစ်ပြီး အဓိကအားဖြင့် ရေမြေဖြည့်စွမ်းမြေဩဇာအဖြစ် စိုက်ပျိုးကြသည်။ ဝါသည် မိုးရာသီတွင် အဓိကစိုက်ပျိုးပြီး မြေအမျိုးအစားများစွာတွင် ပေါက်ရောက်နိုင်သည်။', 5, 11, NULL, NULL, '2025-06-20 00:42:00', '2025-06-20 00:42:00'),
(8, 'မြက်မုံညင်းအဝါက ဘယ်လိုပေါင်းပင်လဲ', 'မြက်မုံညင်းအဝါသည် တစ်နှစ်ခံ ပေါင်းပင်တစ်မျိုးဖြစ်ပြီး Cyperus iria ဟု သိပ္ပံအမည်ဖြင့် ခေါ်ပါသည်။ ၎င်းသည် စပါးစိုက်ခင်းများတွင် အတွေ့ရများသော ပေါင်းပင်ဖြစ်သည်။', 1, 103, 41, NULL, '2025-06-20 00:42:00', '2025-06-20 00:42:00'),
(9, 'မြက်မုံညင်းအဝါရဲ့ လက္ခဏာရပ်တွေက ဘာတွေလဲ', 'မြက်မုံညင်းအဝါ၏ လက္ခဏာရပ်များမှာ တစ်နှစ်ခံပေါင်းပင်ဖြစ်ပြီး ပင်ငယ်၏ထိပ်ဖျားသည် လှံသွားသကဲ့သို့ ချွန်နေခြင်း၊ ပန်းပွင့်များက သေးငယ်ပြီး အဝါရောင်နှင့် အညိုရောင်ဖြစ်ခြင်းတို့ဖြစ်သည်။ စိုထိုင်းသောမြေစေးနှင့် သဲမြေတွင် ကောင်းစွာရှင်သန်နိုင်သည်။', 1, 103, 41, NULL, '2025-06-20 00:42:00', '2025-06-20 00:42:00');


-- Dash Cards
INSERT INTO `dash_cards` (`id`, `card_link`, `image_path`, `card_label`, `created_at`, `updated_at`) VALUES
(1, 'https://www.doa.gov.mm', 'style_images/card1.png', 'စိုက်ပျိုးရေးကဏ္ဍစုံလင် DOA Website', '2025-06-20 00:42:00', '2025-06-20 00:42:00'),
(2, 'https://www.doa.gov.mm/fes', 'style_images/card2.png', 'စိုက်ပျိုးရေး နည်းပညာများ', '2025-06-20 00:42:00', '2025-06-20 00:42:00'),
(3, 'https://www.doa.gov.mm/mis_market/index.php?route=market/weekly_crop_price_by_seller&filter_seller=4', 'style_images/card3.png', 'စိုက်ပျိုးရေးစျေးကွက် သတင်းအချက်အလက်များ', '2025-06-20 00:42:00', '2025-06-20 00:42:00'),
(4, 'https://www.doa.gov.mm/elibrary_new', 'style_images/card4.png', 'စာအုပ်များဖတ်ရှုရန် e-library', '2025-06-20 00:42:00', '2025-06-20 00:42:00');

-- Options (from your provided data, corrected linkedAnswer_id values based on answers table)
INSERT INTO `options` (`id`, `name`, `answer_id`, `linkedAnswer_id`, `created_at`, `updated_at`) VALUES
(1, 'စပါးမိတ်ဆက်', 1, 2, '2025-05-03 00:31:24', '2025-05-09 19:54:34'),
(2, 'စပါးစိုက်စနစ်', 1, 4, '2025-05-03 00:31:33', '2025-05-09 19:54:55'),
(3, 'စပါးအခြောက်တိုက်ရိုက်မျိုးစေ့ချ', 1, 5, '2025-06-20 12:00:00', '2025-06-20 12:00:00'),
-- Adding an option for 'မိတ်ဆက်' (answer_id 3) to demonstrate a linked answer
(4, 'ဝါမိတ်ဆက်', 3, 7, '2025-06-20 12:05:00', '2025-06-20 12:05:00');
-- The example for Coffee introduction was removed as there's no corresponding answer ID for it yet.
-- Add more options as needed, ensuring linkedAnswer_id maps to existing IDs in the answers table.

COMMIT;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
