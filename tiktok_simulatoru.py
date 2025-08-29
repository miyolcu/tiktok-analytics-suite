#!/usr/bin/env python3
"""
🎯 Production-Grade TikTok Data Simulator
128GB RAM + Xeon E5-2680 v4 Optimized Version
Generates 1M videos, 80K users with full interaction networks
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
from typing import Dict, List, Tuple, Optional


class ContentType:
    ORGANIC = "organic"
    BOT_BOOSTED = "bot_boosted"
    MISINFORMATION = "misinformation"
    DISINFORMATION = "disinformation"
    ABUSE = "abuse"
    COORDINATED = "coordinated"


class UserType:
    GENUINE = "genuine"
    BOT = "bot"
    INFLUENCER = "influencer"
    TROLL = "troll"
    COORDINATED_ACTOR = "coordinated_actor"


class InteractionType:
    LIKE = "like"
    COMMENT = "comment"
    SHARE = "share"
    FOLLOW = "follow"
    REPORT = "report"


class ProductionTikTokSimulator:
    """
    Production-grade TikTok simulator optimized for high-performance systems
    Generates realistic manipulation scenarios with user interactions
    """

    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

        # Performance settings for 128GB RAM + Xeon
        self.batch_size = 100000  # Process in large batches
        self.enable_parallel = True
        self.cpu_cores = min(14, mp.cpu_count())  # Xeon E5-2680 v4 has 14 cores

        print(f"🚀 Production TikTok Simulator initialized")
        print(f"   💾 RAM Optimized for: 128GB")
        print(f"   🖥️  CPU Cores available: {self.cpu_cores}")
        print(f"   ⚡ Batch size: {self.batch_size:,}")

        # Enhanced category profiles with manipulation susceptibility
        self.category_profiles = {
            'komedi': {'base_rate': 0.22, 'viral_chance': 0.035, 'manipulation_target': 0.4, 'bot_appeal': 0.6},
            'pet': {'base_rate': 0.20, 'viral_chance': 0.040, 'manipulation_target': 0.2, 'bot_appeal': 0.9},
            'dans': {'base_rate': 0.19, 'viral_chance': 0.030, 'manipulation_target': 0.5, 'bot_appeal': 0.8},
            'eğlence': {'base_rate': 0.18, 'viral_chance': 0.025, 'manipulation_target': 0.4, 'bot_appeal': 0.6},
            'spor': {'base_rate': 0.17, 'viral_chance': 0.025, 'manipulation_target': 0.3, 'bot_appeal': 0.4},
            'müzik': {'base_rate': 0.16, 'viral_chance': 0.028, 'manipulation_target': 0.3, 'bot_appeal': 0.7},
            'gaming': {'base_rate': 0.16, 'viral_chance': 0.022, 'manipulation_target': 0.6, 'bot_appeal': 0.5},
            'güzellik': {'base_rate': 0.15, 'viral_chance': 0.018, 'manipulation_target': 0.8, 'bot_appeal': 0.9},
            'moda': {'base_rate': 0.14, 'viral_chance': 0.015, 'manipulation_target': 0.9, 'bot_appeal': 0.8},
            'seyahat': {'base_rate': 0.13, 'viral_chance': 0.020, 'manipulation_target': 0.3, 'bot_appeal': 0.3},
            'yemek': {'base_rate': 0.12, 'viral_chance': 0.012, 'manipulation_target': 0.2, 'bot_appeal': 0.4},
            'teknoloji': {'base_rate': 0.11, 'viral_chance': 0.015, 'manipulation_target': 0.7, 'bot_appeal': 0.3},
            'sanat': {'base_rate': 0.10, 'viral_chance': 0.012, 'manipulation_target': 0.2, 'bot_appeal': 0.3},
            'diy': {'base_rate': 0.09, 'viral_chance': 0.010, 'manipulation_target': 0.1, 'bot_appeal': 0.2},
            'eğitim': {'base_rate': 0.08, 'viral_chance': 0.008, 'manipulation_target': 0.8, 'bot_appeal': 0.2}
        }

        # Enhanced manipulation campaigns
        self.manipulation_campaigns = {
            'crypto_pump': {
                'duration_days': (3, 14),
                'target_categories': ['teknoloji', 'eğitim'],
                'intensity_range': (0.15, 0.40),
                'bot_coordination': 0.9,
                'content_type': ContentType.DISINFORMATION,
                'share_amplification': (8, 25),
                'fake_engagement_boost': (3, 8)
            },
            'beauty_astroturfing': {
                'duration_days': (21, 60),
                'target_categories': ['güzellik', 'moda'],
                'intensity_range': (0.08, 0.25),
                'bot_coordination': 0.7,
                'content_type': ContentType.BOT_BOOSTED,
                'share_amplification': (2, 6),
                'fake_engagement_boost': (5, 15)
            },
            'health_misinfo': {
                'duration_days': (7, 21),
                'target_categories': ['eğitim', 'güzellik', 'spor'],
                'intensity_range': (0.05, 0.20),
                'bot_coordination': 0.95,
                'content_type': ContentType.MISINFORMATION,
                'share_amplification': (12, 40),
                'fake_engagement_boost': (1, 4)
            },
            'gaming_toxicity': {
                'duration_days': (2, 10),
                'target_categories': ['gaming', 'eğlence'],
                'intensity_range': (0.10, 0.30),
                'bot_coordination': 0.8,
                'content_type': ContentType.COORDINATED,
                'share_amplification': (3, 12),
                'fake_engagement_boost': (2, 6)
            },
            'political_polarization': {
                'duration_days': (5, 30),
                'target_categories': ['eğitim', 'teknoloji', 'eğlence'],
                'intensity_range': (0.12, 0.35),
                'bot_coordination': 0.85,
                'content_type': ContentType.DISINFORMATION,
                'share_amplification': (10, 30),
                'fake_engagement_boost': (4, 12)
            }
        }

        # Temporal patterns for 14-day window
        self.daily_activity_pattern = self._generate_daily_patterns()
        self.hourly_engagement = {
            **{h: 0.2 for h in range(0, 6)},  # Late night: very low
            **{h: 0.5 for h in range(6, 9)},  # Morning: medium
            **{h: 0.8 for h in range(9, 12)},  # Mid-morning: high
            **{h: 1.0 for h in range(12, 14)},  # Lunch: peak
            **{h: 0.9 for h in range(14, 18)},  # Afternoon: high
            **{h: 1.3 for h in range(18, 22)},  # Evening: super peak
            **{h: 0.6 for h in range(22, 24)}  # Night: medium
        }

    def _generate_daily_patterns(self) -> Dict[int, float]:
        """14 günlük daily activity pattern oluştur"""
        # Weekend boost, campaign days, organic trends
        base_pattern = {}
        for day in range(14):
            day_of_week = (datetime.now() - timedelta(days=day)).weekday()

            # Weekend boost (Friday, Saturday, Sunday)
            if day_of_week >= 4:  # Friday=4, Saturday=5, Sunday=6
                multiplier = 1.4
            elif day_of_week == 3:  # Thursday
                multiplier = 1.2
            else:
                multiplier = 1.0

            # Add some randomness
            multiplier *= np.random.uniform(0.85, 1.15)
            base_pattern[day] = multiplier

        return base_pattern

    def generate_users_vectorized(self, user_count=80000, bot_ratio=0.18):
        """Vectorized user generation for performance"""
        print(f"👥 Generating {user_count:,} users (optimized for Xeon)...")

        bot_count = int(user_count * bot_ratio)
        genuine_count = user_count - bot_count

        # Pre-allocate arrays for performance
        user_data = {}

        # Generate IDs
        user_data['user_id'] = [f'user_{i:05d}' for i in range(user_count)]
        user_data['username'] = [f'user_{i:05d}_{random.randint(100, 999)}' for i in range(user_count)]

        # Vectorized user types
        user_types = ['bot'] * bot_count + ['genuine'] * int(genuine_count * 0.87) + \
                     ['influencer'] * int(genuine_count * 0.02) + \
                     ['troll'] * int(genuine_count * 0.06) + \
                     ['coordinated_actor'] * (genuine_count - int(genuine_count * 0.95))

        np.random.shuffle(user_types)
        user_data['user_type'] = user_types

        # Vectorized follower counts (lognormal distribution)
        followers = np.random.lognormal(mean=4.5, sigma=2.2, size=user_count).astype(int)

        # Boost influencers
        influencer_mask = np.array(user_types) == 'influencer'
        followers[influencer_mask] = np.maximum(followers[influencer_mask], 50000)

        user_data['follower_count'] = followers
        user_data['following_count'] = np.random.lognormal(mean=4.0, sigma=1.8, size=user_count).astype(int)

        # Countries (vectorized)
        countries = ['TR', 'US', 'GB', 'DE', 'FR', 'BR', 'IN']
        country_probs = [0.35, 0.20, 0.12, 0.10, 0.08, 0.08, 0.07]
        user_data['country'] = np.random.choice(countries, size=user_count, p=country_probs)

        # Age groups
        age_groups = ['13-17', '18-24', '25-34', '35-44', '45+']
        age_probs = [0.25, 0.42, 0.23, 0.08, 0.02]
        user_data['age_group'] = np.random.choice(age_groups, size=user_count, p=age_probs)

        # Account creation dates (vectorized)
        days_back = np.random.exponential(180, size=user_count)
        days_back = np.minimum(days_back, 1095).astype(int)  # Max 3 years
        creation_dates = [(datetime.now() - timedelta(days=int(d))).date() for d in days_back]
        user_data['account_creation_date'] = creation_dates

        # Verification status
        verification_prob = np.where(followers < 10000, 0.01,
                                     np.where(followers < 100000, 0.12, 0.50))
        user_data['is_verified'] = np.random.random(user_count) < verification_prob

        # Bot-specific attributes
        bot_mask = np.array(user_types) == 'bot'
        bot_cluster_ids = np.full(user_count, -1)

        if bot_count > 0:
            n_clusters = max(1, bot_count // 75)  # 75 bots per cluster
            bot_clusters = np.random.randint(0, n_clusters, size=bot_count)
            bot_cluster_ids[bot_mask] = bot_clusters

        user_data['bot_cluster_id'] = bot_cluster_ids
        user_data['coordination_strength'] = np.where(bot_mask, np.random.uniform(0.6, 0.95, user_count), 0.0)

        users_df = pd.DataFrame(user_data)
        users_df['is_creator'] = users_df['follower_count'] > 1000

        print(f"✅ Generated {len(users_df):,} users in vectorized mode")
        print(f"   • Genuine: {len(users_df[users_df['user_type'] == 'genuine']):,}")
        print(f"   • Bots: {len(users_df[users_df['user_type'] == 'bot']):,}")
        print(f"   • Bot clusters: {len(users_df[users_df['bot_cluster_id'] >= 0]['bot_cluster_id'].unique())}")
        print(f"   • Influencers: {len(users_df[users_df['user_type'] == 'influencer']):,}")
        print(f"   • Verified: {users_df['is_verified'].sum():,}")

        return users_df

    def generate_manipulation_campaigns(self) -> List[Dict]:
        """Enhanced manipulation campaigns for 14-day period"""
        campaigns = []

        # Generate 3-7 overlapping campaigns
        n_campaigns = np.random.randint(3, 8)

        for i in range(n_campaigns):
            campaign_type = np.random.choice(list(self.manipulation_campaigns.keys()))
            config = self.manipulation_campaigns[campaign_type]

            # Campaign timing within 14-day window
            duration = np.random.randint(*config['duration_days'])

            # Start date within the 14-day window
            max_start_day = min(14 - duration, 13)
            start_day = np.random.randint(0, max(1, max_start_day + 1))

            start_date = datetime.now() - timedelta(days=start_day)
            end_date = start_date + timedelta(days=duration)

            intensity = np.random.uniform(*config['intensity_range'])

            campaign = {
                'id': i,
                'type': campaign_type,
                'start_date': start_date,
                'end_date': end_date,
                'duration_days': duration,
                'intensity': intensity,
                'config': config,
                'active_day_range': (start_day, start_day + duration),
                'bot_coordination': config['bot_coordination'],
                'target_categories': config['target_categories']
            }
            campaigns.append(campaign)

        print(f"📋 Generated {len(campaigns)} manipulation campaigns:")
        for camp in campaigns:
            days_active = (camp['end_date'] - camp['start_date']).days
            print(f"   • {camp['type']}: {days_active}d duration, {camp['intensity']:.1%} intensity")

        return campaigns

    def generate_videos_parallel(self, video_count=1000000, users_df=None):
        """Parallel video generation for 1M videos"""
        if users_df is None:
            raise ValueError("Users DataFrame required")

        print(f"🎬 Generating {video_count:,} videos with parallel processing...")

        # Generate campaigns for the 14-day period
        campaigns = self.generate_manipulation_campaigns()

        # Split work across CPU cores
        videos_per_core = video_count // self.cpu_cores
        remaining_videos = video_count % self.cpu_cores

        chunk_sizes = [videos_per_core] * self.cpu_cores
        chunk_sizes[0] += remaining_videos  # Add remainder to first chunk

        print(f"   🖥️  Using {self.cpu_cores} cores")
        print(f"   📦 Chunk sizes: {chunk_sizes}")

        # Prepare arguments for parallel processing
        args_list = []
        video_id_start = 0

        for core_id, chunk_size in enumerate(chunk_sizes):
            args = (
                core_id, chunk_size, video_id_start,
                users_df, campaigns,
                self.category_profiles, self.daily_activity_pattern, self.hourly_engagement
            )
            args_list.append(args)
            video_id_start += chunk_size

        # Process in parallel
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=self.cpu_cores) as executor:
            chunk_results = list(executor.map(self._generate_video_chunk, args_list))

        # Combine results
        all_videos = []
        for chunk_videos in chunk_results:
            all_videos.extend(chunk_videos)

        videos_df = pd.DataFrame(all_videos)

        generation_time = time.time() - start_time
        print(f"✅ Generated {len(videos_df):,} videos in {generation_time / 60:.2f} minutes")
        print(f"   ⚡ Performance: {len(videos_df) / (generation_time * 1000):.1f}K videos/second")

        return videos_df

    @staticmethod
    def _generate_video_chunk(args) -> List[Dict]:
        """Static method for parallel video chunk generation"""
        core_id, chunk_size, video_id_start, users_df, campaigns, category_profiles, daily_patterns, hourly_patterns = args

        print(f"      Core {core_id}: Generating {chunk_size:,} videos...")

        videos = []
        current_time = datetime.now()

        for i in range(chunk_size):
            video_id = video_id_start + i

            # Select user
            user = users_df.sample(1).iloc[0]

            # Determine content type based on campaigns and randomness
            content_type, campaign = ProductionTikTokSimulator._determine_content_type_static(
                campaigns, current_time, category_profiles
            )

            # Generate video with manipulation
            video = ProductionTikTokSimulator._generate_video_static(
                video_id, user, content_type, campaign, category_profiles,
                daily_patterns, hourly_patterns
            )

            videos.append(video)

            # Progress update every 50K videos
            if (i + 1) % 50000 == 0:
                print(f"         Core {core_id}: {i + 1:,}/{chunk_size:,} completed")

        print(f"      ✅ Core {core_id}: Completed {chunk_size:,} videos")
        return videos

    @staticmethod
    def _determine_content_type_static(campaigns, current_time, category_profiles):
        """Static content type determination for parallel processing"""

        # Check active campaigns
        active_campaigns = [c for c in campaigns
                            if c['start_date'] <= current_time <= c['end_date']]

        # Base probability for normal content
        base_prob = np.random.random()

        # 75% chance for normal distribution even with active campaigns
        if not active_campaigns or base_prob < 0.75:
            if base_prob < 0.60:  # 60% organic
                return ContentType.ORGANIC, None
            elif base_prob < 0.68:  # 8% bot_boosted
                return ContentType.BOT_BOOSTED, None
            elif base_prob < 0.73:  # 5% abuse
                return ContentType.ABUSE, None
            elif base_prob < 0.75:  # 2% misinformation
                return ContentType.MISINFORMATION, None

        # Campaign influence (25% chance)
        if active_campaigns and base_prob >= 0.75:
            campaign = np.random.choice(active_campaigns)
            if np.random.random() < campaign['intensity']:
                return campaign['config']['content_type'], campaign

        return ContentType.ORGANIC, None

    @staticmethod
    def _generate_video_static(video_id, user, content_type, campaign,
                               category_profiles, daily_patterns, hourly_patterns):
        """Static video generation for parallel processing"""

        # Category selection
        if campaign and np.random.random() < 0.7:
            category = np.random.choice(campaign['target_categories'])
        else:
            categories = list(category_profiles.keys())
            category = np.random.choice(categories)

        profile = category_profiles[category]

        # Time generation - last 14 days
        days_ago = min(int(np.random.exponential(4)), 14)
        day_multiplier = daily_patterns.get(days_ago, 1.0)

        hour = np.random.choice(24, p=[hourly_patterns[h] / sum(hourly_patterns.values()) for h in range(24)])
        hour_multiplier = hourly_patterns[hour]

        post_time = datetime.now() - timedelta(days=days_ago, hours=hour)

        # Base metrics calculation
        user_influence = max(1, np.log10(user['follower_count'] + 1))
        time_boost = day_multiplier * hour_multiplier

        base_views = int(np.random.lognormal(mean=8.5, sigma=2.0) * user_influence * time_boost)

        # Content-type specific manipulation
        if content_type == ContentType.ORGANIC:
            like_rate = np.random.uniform(profile['base_rate'] * 0.8, profile['base_rate'] * 1.2)
            comment_rate = np.random.uniform(0.005, 0.025)
            share_rate = np.random.uniform(0.01, 0.04)
            views = base_views

            # Viral chance
            if np.random.random() < profile['viral_chance']:
                views *= np.random.randint(8, 50)

        elif content_type == ContentType.BOT_BOOSTED:
            like_rate = np.random.uniform(0.30, 0.65)  # Anormal yüksek
            comment_rate = np.random.uniform(0.001, 0.008)  # Düşük
            share_rate = np.random.uniform(0.02, 0.08)
            fake_boost = np.random.uniform(3, 10) if campaign else np.random.uniform(2, 5)
            views = int(base_views * fake_boost)

        elif content_type == ContentType.MISINFORMATION:
            like_rate = np.random.uniform(profile['base_rate'] * 0.9, profile['base_rate'] * 1.3)
            comment_rate = np.random.uniform(0.001, 0.006)  # Çok düşük - kimse tartışmak istemiyor
            share_rate = np.random.uniform(0.20, 0.50)  # ÇOK YÜKSEK - viral misinformation
            views = int(base_views * np.random.uniform(4, 15))

        elif content_type == ContentType.DISINFORMATION:
            like_rate = np.random.uniform(0.15, 0.35)
            comment_rate = np.random.uniform(0.002, 0.012)
            share_rate = np.random.uniform(0.15, 0.35)  # Yüksek share
            views = int(base_views * np.random.uniform(6, 20))

        elif content_type == ContentType.COORDINATED:
            like_rate = np.random.uniform(0.25, 0.45)
            comment_rate = np.random.uniform(0.03, 0.08)  # Yüksek comment (harassment)
            share_rate = np.random.uniform(0.05, 0.15)
            views = int(base_views * np.random.uniform(3, 12))

        else:  # ABUSE
            like_rate = np.random.uniform(0.01, 0.04)  # Platform suppressed
            comment_rate = np.random.uniform(0.008, 0.025)  # Reports/complaints
            share_rate = np.random.uniform(0.001, 0.01)
            views = int(base_views * np.random.uniform(0.1, 0.3))

        # Calculate final metrics
        likes = int(views * like_rate)
        comments = int(views * comment_rate)
        shares = int(views * share_rate)
        engagement_rate = (likes + comments + shares) / max(views, 1) * 100

        # Anomaly detection metrics
        like_comment_ratio = likes / max(comments, 1)
        share_like_ratio = shares / max(likes, 1)

        # Simple anomaly score
        anomaly_score = 0.0
        if like_comment_ratio > 150: anomaly_score += 0.3
        if share_like_ratio > 0.4: anomaly_score += 0.4
        if engagement_rate > 30: anomaly_score += 0.2
        if content_type != ContentType.ORGANIC: anomaly_score += 0.15
        anomaly_score = min(1.0, anomaly_score)

        return {
            'video_id': f'video_{video_id:07d}',
            'user_id': user['user_id'],
            'creator_username': user['username'],
            'creator_type': user['user_type'],
            'creator_followers': user['follower_count'],
            'creator_verified': user['is_verified'],
            'category': category,
            'content_type': content_type.value if hasattr(content_type, 'value') else content_type,
            'campaign_id': campaign['id'] if campaign else None,
            'campaign_type': campaign['type'] if campaign else None,
            'post_datetime': post_time,
            'post_date': post_time.date(),
            'post_hour': hour,
            'days_ago': days_ago,
            'views': views,
            'likes': likes,
            'comments': comments,
            'shares': shares,
            'engagement_rate': engagement_rate,
            'is_viral': views > 500000,  # 500K threshold for 1M dataset
            'like_comment_ratio': like_comment_ratio,
            'share_like_ratio': share_like_ratio,
            'anomaly_score': anomaly_score,
            'is_suspicious': anomaly_score > 0.6,
            'bot_cluster_id': user.get('bot_cluster_id', -1),
            'time_multiplier': time_boost,
            'duration_seconds': np.random.choice([15, 30, 60, 120], p=[0.5, 0.3, 0.15, 0.05])
        }

    def generate_user_interactions_parallel(self, users_df, videos_df, sample_ratio=0.5):
        """PARALLEL user-to-user interactions - 14 cores unleashed! 🚀"""
        print(f"🤝 Generating user interactions (PARALLEL MODE)...")

        # With 128GB RAM, we can process much more videos!
        sample_size = int(len(videos_df) * sample_ratio)  # 50% of 1M = 500K videos
        sampled_videos = videos_df.sample(n=sample_size).reset_index(drop=True)

        print(f"   🚀 Processing {len(sampled_videos):,} videos with {self.cpu_cores} cores")
        print(f"   💾 Estimated interactions: ~{sample_size * 3:,} (avg 3 per video)")

        # Split videos across cores
        videos_per_core = len(sampled_videos) // self.cpu_cores
        remaining_videos = len(sampled_videos) % self.cpu_cores

        chunk_sizes = [videos_per_core] * self.cpu_cores
        chunk_sizes[0] += remaining_videos

        print(f"   📦 Video chunks per core: {chunk_sizes}")

        # Prepare arguments for parallel processing
        args_list = []
        video_start_idx = 0
        interaction_id_start = 0

        for core_id, chunk_size in enumerate(chunk_sizes):
            video_chunk = sampled_videos.iloc[video_start_idx:video_start_idx + chunk_size]

            args = (
                core_id, video_chunk, users_df,
                interaction_id_start, f"Core_{core_id}"
            )
            args_list.append(args)

            video_start_idx += chunk_size
            interaction_id_start += chunk_size * 5  # Estimated 5 interactions per video max

        # PARALLEL PROCESSING MAGIC! ⚡
        start_time = time.time()
        print(f"   ⚡ Starting parallel interaction generation at {datetime.now().strftime('%H:%M:%S')}")

        with ProcessPoolExecutor(max_workers=self.cpu_cores) as executor:
            chunk_results = list(executor.map(self._generate_interaction_chunk, args_list))

        # Combine all interactions
        all_interactions = []
        for chunk_interactions in chunk_results:
            all_interactions.extend(chunk_interactions)

        interactions_df = pd.DataFrame(all_interactions)

        interaction_time = time.time() - start_time

        print(f"✅ Generated {len(interactions_df):,} interactions in {interaction_time:.1f}s")
        print(f"   ⚡ Performance: {len(interactions_df) / interaction_time:.0f} interactions/second")
        print(f"   🎯 Breakdown:")
        print(f"      • Likes: {len(interactions_df[interactions_df['interaction_type'] == 'like']):,}")
        print(f"      • Comments: {len(interactions_df[interactions_df['interaction_type'] == 'comment']):,}")
        print(f"      • Shares: {len(interactions_df[interactions_df['interaction_type'] == 'share']):,}")
        print(f"      • Reports: {len(interactions_df[interactions_df['interaction_type'] == 'report']):,}")
        print(f"      • Bot interactions: {interactions_df['is_bot_interaction'].sum():,}")

        return interactions_df

    @staticmethod
    def _generate_interaction_chunk(args) -> List[Dict]:
        """Static method for parallel interaction generation"""
        core_id, video_chunk, users_df, interaction_id_start, core_name = args

        print(f"         {core_name}: Processing {len(video_chunk):,} videos...")

        interactions = []
        interaction_id = interaction_id_start

        for idx, (_, video) in enumerate(video_chunk.iterrows()):
            # Progress update every 10K videos
            if (idx + 1) % 10000 == 0:
                print(f"            {core_name}: {idx + 1:,}/{len(video_chunk):,} videos processed")

            video_engagement = video['engagement_rate']
            video_content_type = video['content_type']

            # Smart interaction count based on content and engagement
            if video_content_type in ['misinformation', 'disinformation']:
                # Misinformation gets more shares, fewer comments
                base_interactions = np.random.poisson(6)
                share_heavy = True
            elif video_content_type == 'abuse':
                # Abuse gets more reports
                base_interactions = np.random.poisson(3)
                share_heavy = False
            elif video_engagement > 30:  # Very high engagement
                base_interactions = np.random.poisson(12)
                share_heavy = False
            elif video_engagement > 20:  # High engagement
                base_interactions = np.random.poisson(8)
                share_heavy = False
            elif video_engagement > 10:  # Medium engagement
                base_interactions = np.random.poisson(5)
                share_heavy = False
            else:  # Low engagement
                base_interactions = np.random.poisson(2)
                share_heavy = False

            # Cap interactions to prevent memory explosion
            n_interactions = min(base_interactions, 20)

            # Generate interactions for this video
            for _ in range(n_interactions):
                # Select interacting user (not the video creator)
                possible_users = users_df[users_df['user_id'] != video['user_id']]
                if len(possible_users) == 0:
                    continue

                # Weighted user selection (bots more active, influencers get more engagement)
                user_weights = np.ones(len(possible_users))

                # Boost bot activity
                bot_mask = possible_users['user_type'] == 'bot'
                user_weights[bot_mask] *= 2.5

                # Boost high-follower users (they're more active)
                high_follower_mask = possible_users['follower_count'] > 10000
                user_weights[high_follower_mask] *= 1.5

                # Normalize weights
                user_weights = user_weights / user_weights.sum()

                # Select user
                selected_idx = np.random.choice(len(possible_users), p=user_weights)
                interacting_user = possible_users.iloc[selected_idx]

                # Determine interaction type based on content and user type
                if video_content_type == 'abuse':
                    if interacting_user['user_type'] == 'bot':
                        interaction_type = np.random.choice(['like', 'report'], p=[0.8, 0.2])
                    else:
                        interaction_type = np.random.choice(['report', 'like'], p=[0.7, 0.3])

                elif video_content_type in ['misinformation', 'disinformation']:
                    if share_heavy:
                        if interacting_user['user_type'] == 'bot':
                            interaction_type = np.random.choice(['share', 'like', 'comment'], p=[0.6, 0.35, 0.05])
                        else:
                            interaction_type = np.random.choice(['share', 'like', 'comment'], p=[0.5, 0.3, 0.2])
                    else:
                        interaction_type = np.random.choice(['like', 'share', 'comment'], p=[0.4, 0.4, 0.2])

                elif video_content_type == 'coordinated':
                    # Coordinated attacks have more comments (harassment)
                    interaction_type = np.random.choice(['comment', 'like', 'share', 'report'], p=[0.4, 0.3, 0.2, 0.1])

                else:  # Organic content
                    if interacting_user['user_type'] == 'bot':
                        interaction_type = np.random.choice(['like', 'comment', 'share'], p=[0.7, 0.15, 0.15])
                    else:
                        interaction_type = np.random.choice(['like', 'comment', 'share'], p=[0.6, 0.25, 0.15])

                # Interaction timing - realistic decay
                video_age_hours = (datetime.now() - video['post_datetime']).total_seconds() / 3600
                max_delay = min(48, video_age_hours)  # Max 48h or video age

                # Exponential decay - most interactions happen early
                if max_delay > 0:
                    delay_hours = np.random.exponential(4)  # Most within 4h
                    delay_hours = min(delay_hours, max_delay)
                else:
                    delay_hours = 0

                interaction_time = video['post_datetime'] + timedelta(hours=delay_hours)

                # Cross-country interaction check
                video_creator = users_df[users_df['user_id'] == video['user_id']]
                if len(video_creator) > 0:
                    creator_country = video_creator.iloc[0]['country']
                    is_cross_country = interacting_user['country'] != creator_country
                else:
                    is_cross_country = False

                # Create interaction record
                interaction = {
                    'interaction_id': f'int_{interaction_id:08d}',
                    'video_id': video['video_id'],
                    'video_creator': video['user_id'],
                    'video_category': video['category'],
                    'video_content_type': video_content_type,
                    'video_engagement_rate': video_engagement,
                    'video_views': video['views'],
                    'interacting_user': interacting_user['user_id'],
                    'interacting_user_type': interacting_user['user_type'],
                    'interacting_user_followers': interacting_user['follower_count'],
                    'interacting_user_country': interacting_user['country'],
                    'interaction_type': interaction_type,
                    'interaction_datetime': interaction_time,
                    'delay_hours': delay_hours,
                    'is_bot_interaction': interacting_user['user_type'] == 'bot',
                    'is_cross_country': is_cross_country,
                    'is_verified_user': interacting_user['is_verified'],
                    'bot_cluster_id': interacting_user.get('bot_cluster_id', -1)
                }

                interactions.append(interaction)
                interaction_id += 1

        print(f"         ✅ {core_name}: Generated {len(interactions):,} interactions")
        return interactions


def save_production_data(users_df, videos_df, interactions_df=None):
    """Save production data to optimized files"""
    output_dir = 'tiktok_production_data'
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"\n💾 SAVING PRODUCTION DATA...")
    print(f"📁 Output directory: {output_dir}/")

    file_info = {}

    # 1. Users CSV
    print(f"   📄 Saving users data...")
    users_path = os.path.join(output_dir, f'users_{timestamp}.csv')
    users_df.to_csv(users_path, index=False, encoding='utf-8')
    users_size = os.path.getsize(users_path) / (1024 * 1024)
    file_info['users'] = {'path': users_path, 'size_mb': users_size, 'rows': len(users_df)}
    print(f"      ✅ Users: {len(users_df):,} rows → {users_size:.1f} MB")

    # 2. Videos CSV (chunked for large files)
    print(f"   📄 Saving videos data...")
    videos_path = os.path.join(output_dir, f'videos_{timestamp}.csv')

    chunk_size = 100000  # 100K rows per chunk for better performance
    total_chunks = (len(videos_df) + chunk_size - 1) // chunk_size

    print(f"      💾 Processing {total_chunks} chunks...")
    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(videos_df))
        chunk = videos_df.iloc[start_idx:end_idx]

        mode = 'w' if i == 0 else 'a'
        header = i == 0

        chunk.to_csv(videos_path, mode=mode, header=header, index=False, encoding='utf-8')

        if (i + 1) % 5 == 0 or i == total_chunks - 1:
            print(f"         Chunk {i + 1}/{total_chunks} saved ({end_idx - start_idx:,} rows)")

    videos_size = os.path.getsize(videos_path) / (1024 * 1024)
    file_info['videos'] = {'path': videos_path, 'size_mb': videos_size, 'rows': len(videos_df)}
    print(f"      ✅ Videos: {len(videos_df):,} rows → {videos_size:.1f} MB")

    # 3. Interactions CSV (if available)
    if interactions_df is not None and len(interactions_df) > 0:
        print(f"   📄 Saving interactions data...")
        interactions_path = os.path.join(output_dir, f'interactions_{timestamp}.csv')
        interactions_df.to_csv(interactions_path, index=False, encoding='utf-8')
        interactions_size = os.path.getsize(interactions_path) / (1024 * 1024)
        file_info['interactions'] = {'path': interactions_path, 'size_mb': interactions_size,
                                     'rows': len(interactions_df)}
        print(f"      ✅ Interactions: {len(interactions_df):,} rows → {interactions_size:.1f} MB")

    # 4. Summary JSON
    print(f"   📄 Saving summary data...")
    summary_path = os.path.join(output_dir, f'summary_{timestamp}.json')

    summary_data = {
        'metadata': {
            'generation_timestamp': timestamp,
            'system_specs': '128GB RAM + Intel Xeon E5-2680 v4',
            'total_users': len(users_df),
            'total_videos': len(videos_df),
            'total_interactions': len(interactions_df) if interactions_df is not None else 0,
            'time_range_days': 14,
            'date_range': {
                'start': str(videos_df['post_date'].min()),
                'end': str(videos_df['post_date'].max())
            }
        },
        'quick_stats': {
            'content_types': videos_df['content_type'].value_counts().to_dict(),
            'user_types': users_df['user_type'].value_counts().to_dict(),
            'categories': videos_df['category'].value_counts().to_dict(),
            'countries': users_df['country'].value_counts().to_dict(),
            'engagement_stats': {
                'avg_engagement_rate': float(videos_df['engagement_rate'].mean()),
                'total_views': int(videos_df['views'].sum()),
                'total_likes': int(videos_df['likes'].sum()),
                'total_shares': int(videos_df['shares'].sum()),
                'viral_videos': int(videos_df['is_viral'].sum()),
                'suspicious_content': int(videos_df['is_suspicious'].sum())
            }
        },
        'manipulation_metrics': {
            'suspicious_content_ratio': float(videos_df['is_suspicious'].mean()),
            'avg_anomaly_score': float(videos_df['anomaly_score'].mean()),
            'bot_created_content': int(len(videos_df[videos_df['creator_type'] == 'bot'])),
            'active_campaigns': int(videos_df['campaign_id'].nunique()),
            'high_anomaly_videos': int(len(videos_df[videos_df['anomaly_score'] > 0.8]))
        },
        'files': file_info
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    summary_size = os.path.getsize(summary_path) / (1024 * 1024)
    file_info['summary'] = {'path': summary_path, 'size_mb': summary_size}

    total_size = sum(info['size_mb'] for info in file_info.values())

    print(f"\n📊 FILE SUMMARY:")
    for file_type, info in file_info.items():
        if 'rows' in info:
            print(f"   • {file_type.upper()}: {info['size_mb']:.1f} MB ({info['rows']:,} rows)")
        else:
            print(f"   • {file_type.upper()}: {info['size_mb']:.1f} MB")
    print(f"   📦 TOTAL: {total_size:.1f} MB")

    return file_info


def generate_production_dataset():
    """Generate complete production dataset optimized for high-end hardware"""

    print("🏭 PRODUCTION DATASET GENERATION")
    print("=" * 60)
    print("🖥️  System: 128GB RAM + Intel Xeon E5-2680 v4 (14 cores)")
    print("🎯 Target: 1M videos, 80K users, 14 days, 600K video interactions")
    print("⚡ Estimated time: 3-5 minutes (FULLY PARALLEL)")
    print("💾 Estimated output: 500-800 MB (more interactions!)")
    print("=" * 60)

    simulator = ProductionTikTokSimulator()
    start_time = time.time()

    # Step 1: Generate users (vectorized)
    print(f"\n🚀 STEP 1: User Generation")
    users_df = simulator.generate_users_vectorized(user_count=80000, bot_ratio=0.18)
    user_time = time.time() - start_time
    print(f"   ⏱️ Users completed in {user_time:.1f}s")

    # Step 2: Generate videos (parallel)
    print(f"\n🚀 STEP 2: Video Generation (Parallel)")
    video_start = time.time()
    videos_df = simulator.generate_videos_parallel(video_count=1000000, users_df=users_df)
    video_time = time.time() - video_start
    print(f"   ⏱️ Videos completed in {video_time / 60:.2f} minutes")

    # Step 3: Generate interactions (PARALLEL!)
    print(f"\n🚀 STEP 3: User Interactions (14 CORES UNLEASHED!)")
    interactions_start = time.time()
    interactions_df = simulator.generate_user_interactions_parallel(users_df, videos_df,
                                                                    sample_ratio=0.6)  # 60% of 1M videos!
    interactions_time = time.time() - interactions_start
    sample_video_count = int(len(videos_df) * 0.6)  # For stats calculation
    print(f"   ⏱️ Interactions completed in {interactions_time / 60:.2f} minutes")

    # Step 4: Save all data
    print(f"\n🚀 STEP 4: Data Export")
    save_start = time.time()
    file_info = save_production_data(users_df, videos_df, interactions_df)
    save_time = time.time() - save_start

    total_time = time.time() - start_time

    print(f"\n🎉 PRODUCTION DATASET COMPLETED!")
    print(f"⏱️  Total time: {total_time / 60:.2f} minutes")
    print(f"   • User generation: {user_time:.1f}s")
    print(f"   • Video generation: {video_time / 60:.2f}m")
    print(f"   • Interaction generation: {interactions_time / 60:.2f}m")
    print(f"   • File saving: {save_time:.1f}s")
    print(f"🚀 Performance: {len(videos_df) / (total_time / 60):.0f}K videos/minute")
    print(f"🤝 Interaction Performance: {len(interactions_df) / (interactions_time / 60):.0f}K interactions/minute")

    # Additional stats
    print(f"\n📊 FINAL DATASET STATS:")
    print(f"   • Videos per user: {len(videos_df) / len(users_df):.1f}")
    print(f"   • Interactions per video: {len(interactions_df) / sample_video_count:.1f}")
    print(f"   • Bot interaction ratio: {interactions_df['is_bot_interaction'].mean():.1%}")
    print(f"   • Cross-country interactions: {interactions_df['is_cross_country'].mean():.1%}")
    print(f"   • Videos with high anomaly: {(videos_df['anomaly_score'] > 0.8).sum():,}")

    return {
        'users': users_df,
        'videos': videos_df,
        'interactions': interactions_df,
        'files': file_info,
        'performance': {
            'total_time_minutes': total_time / 60,
            'videos_per_minute': len(videos_df) / (total_time / 60),
            'interactions_per_minute': len(interactions_df) / (interactions_time / 60) if interactions_time > 0 else 0,
            'cores_utilized': simulator.cpu_cores,
            'memory_efficient': True
        }
    }


if __name__ == '__main__':
    # Production dataset generation
    print("🏭 Starting Production TikTok Dataset Generation...")

    results = generate_production_dataset()

    print(f"\n✨ Dataset ready for analysis and dashboard creation!")
    print(f"📁 Files location: tiktok_production_data/")
    print(f"📊 Ready to import into your analysis tools!")

# Export functions
__all__ = [
    'ProductionTikTokSimulator',
    'generate_production_dataset',
    'save_production_data'
]