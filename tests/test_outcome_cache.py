"""
结果感知缓存优化测试 — OATS-Inspired Quality Cache

测试创新点：
1. 质量分 EMA 更新
2. 有效阈值动态调整
3. 结果反馈驱动的质量演化
4. 过期条目清理
"""

import pytest
import tempfile

from task_router.outcome_cache import OutcomeAwareCache, get_outcome_cache


class TestOutcomeAwareCache:
    """结果感知缓存管理器"""

    @pytest.fixture
    def cache_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    @pytest.fixture
    def cache(self, cache_dir):
        return OutcomeAwareCache(cache_dir)

    # ── 质量分基础测试 ──

    def test_default_quality(self, cache):
        """新条目应有默认质量分"""
        q = cache.get_quality("new_key")
        assert q == OutcomeAwareCache.DEFAULT_QUALITY

    def test_quality_range(self, cache):
        """质量分应在 [MIN_QUALITY, MAX_QUALITY] 范围内"""
        # 大量成功
        for _ in range(100):
            cache.record_outcome("key1", True)
        assert OutcomeAwareCache.MIN_QUALITY <= cache.get_quality("key1") <= OutcomeAwareCache.MAX_QUALITY

        # 大量失败
        for _ in range(100):
            cache.record_outcome("key2", False)
        assert OutcomeAwareCache.MIN_QUALITY <= cache.get_quality("key2") <= OutcomeAwareCache.MAX_QUALITY

    def test_success_increases_quality(self, cache):
        """成功结果应提高质量分"""
        initial = cache.get_quality("key")
        for _ in range(10):
            cache.record_outcome("key", True)
        assert cache.get_quality("key") > initial

    def test_failure_decreases_quality(self, cache):
        """失败结果应降低质量分"""
        initial = cache.get_quality("key")
        for _ in range(10):
            cache.record_outcome("key", False)
        assert cache.get_quality("key") < initial

    def test_quality_minimum_bound(self, cache):
        """质量分不应低于 MIN_QUALITY"""
        for _ in range(200):
            cache.record_outcome("key", False)
        assert cache.get_quality("key") >= OutcomeAwareCache.MIN_QUALITY

    def test_quality_maximum_bound(self, cache):
        """质量分不应超过 MAX_QUALITY"""
        for _ in range(200):
            cache.record_outcome("key", True)
        assert cache.get_quality("key") <= OutcomeAwareCache.MAX_QUALITY

    # ── EMA 更新测试 ──

    def test_ema_convergence(self, cache):
        """EMA 应趋向于 reward 值"""
        lr = OutcomeAwareCache.LEARNING_RATE

        # 纯成功 → 趋向 1.0
        quality = OutcomeAwareCache.DEFAULT_QUALITY
        for _ in range(100):
            quality = quality * (1 - lr) + 1.0 * lr
        assert quality > 0.9  # 接近 1.0

        # 纯失败 → 趋向 0.0
        quality = OutcomeAwareCache.DEFAULT_QUALITY
        for _ in range(100):
            quality = quality * (1 - lr) + 0.0 * lr
        assert quality < 0.2  # 接近 0.0（但有 MIN_QUALITY 下界）

    def test_ema_mixed_results(self, cache):
        """混合结果应产生中等质量分"""
        for i in range(100):
            cache.record_outcome("key", i % 2 == 0)  # 50% 成功

        q = cache.get_quality("key")
        # 应该接近 0.5
        assert 0.3 < q < 0.7

    def test_ema_recent_results_weight_more(self, cache):
        """近期结果应有更大权重（EMA 特性）"""
        # 先 100 次失败
        for _ in range(100):
            cache.record_outcome("key", False)

        # 再 100 次成功
        for _ in range(100):
            cache.record_outcome("key", True)

        q_after_success = cache.get_quality("key")

        # 对比：先 100 次成功再 100 次失败
        for _ in range(100):
            cache.record_outcome("key2", True)
        for _ in range(100):
            cache.record_outcome("key2", False)

        q_after_failure = cache.get_quality("key2")

        # 近期成功的质量分应高于近期失败的
        assert q_after_success > q_after_failure

    # ── 有效阈值测试 ──

    def test_effective_threshold_default(self, cache):
        """默认质量分时，阈值应接近 base_threshold"""
        threshold = cache.get_effective_threshold("new_key", base_threshold=0.85)
        assert abs(threshold - 0.85) < 0.01  # DEFAULT_QUALITY = 0.5 → offset = 0

    def test_high_quality_lower_threshold(self, cache):
        """高质量条目应有更低的阈值（更容易命中）"""
        for _ in range(50):
            cache.record_outcome("good_key", True)

        default_threshold = cache.get_effective_threshold("new_key", base_threshold=0.85)
        good_threshold = cache.get_effective_threshold("good_key", base_threshold=0.85)

        assert good_threshold < default_threshold, \
            "高质量条目的阈值应更低（更容易命中缓存）"

    def test_low_quality_higher_threshold(self, cache):
        """低质量条目应有更高的阈值（更难命中）"""
        for _ in range(50):
            cache.record_outcome("bad_key", False)

        default_threshold = cache.get_effective_threshold("new_key", base_threshold=0.85)
        bad_threshold = cache.get_effective_threshold("bad_key", base_threshold=0.85)

        assert bad_threshold > default_threshold, \
            "低质量条目的阈值应更高（更难命中缓存）"

    def test_threshold_bounds(self, cache):
        """阈值应在 [0.5, 0.95] 范围内"""
        # 极高质量
        for _ in range(100):
            cache.record_outcome("good", True)
        t = cache.get_effective_threshold("good", base_threshold=0.5)
        assert t >= 0.5

        # 极低质量
        for _ in range(100):
            cache.record_outcome("bad", False)
        t = cache.get_effective_threshold("bad", base_threshold=0.95)
        assert t <= 0.95

    def test_threshold_different_base(self, cache):
        """不同 base_threshold 应产生不同有效阈值"""
        t1 = cache.get_effective_threshold("key", base_threshold=0.7)
        t2 = cache.get_effective_threshold("key", base_threshold=0.9)
        assert t1 != t2

    # ── 历史记录测试 ──

    def test_outcome_history_limit(self, cache):
        """历史记录应限制在 100 条"""
        for i in range(150):
            cache.record_outcome("key", i % 2 == 0)

        assert len(cache._outcome_history["key"]) == 100

    def test_different_keys_independent(self, cache):
        """不同 key 的质量分应独立"""
        for _ in range(20):
            cache.record_outcome("key_a", True)
            cache.record_outcome("key_b", False)

        assert cache.get_quality("key_a") > cache.get_quality("key_b")

    # ── 持久化测试 ──

    def test_persistence(self, cache_dir):
        """质量模型应持久化"""
        cache1 = OutcomeAwareCache(cache_dir)
        for _ in range(15):  # 超过 10 次才触发保存
            cache1.record_outcome("key", True)

        # 重新加载
        cache2 = OutcomeAwareCache(cache_dir)
        assert cache2.get_quality("key") > OutcomeAwareCache.DEFAULT_QUALITY

    # ── 统计测试 ──

    def test_stats_empty(self, cache):
        """空缓存应返回零统计"""
        stats = cache.get_stats()
        assert stats["total_entries"] == 0

    def test_stats_populated(self, cache):
        """有数据时应返回正确统计"""
        for _ in range(10):
            cache.record_outcome("good", True)
        for _ in range(10):
            cache.record_outcome("bad", False)
        cache.record_outcome("neutral", True)

        stats = cache.get_stats()
        assert stats["total_entries"] == 3
        assert stats["avg_quality"] > 0
        assert "high_quality_count" in stats
        assert "low_quality_count" in stats

    # ── 清理测试 ──

    def test_cleanup_stale(self, cache):
        """应清理低质量条目"""
        for _ in range(50):
            cache.record_outcome("stale", False)

        removed = cache.cleanup_stale(min_quality=0.2)
        assert removed >= 1
        assert cache.get_quality("stale") == OutcomeAwareCache.DEFAULT_QUALITY  # 已被清理，返回默认值

    def test_cleanup_preserves_good(self, cache):
        """清理不应影响高质量条目"""
        for _ in range(10):
            cache.record_outcome("good", True)

        cache.cleanup_stale(min_quality=0.2)
        assert cache.get_quality("good") > OutcomeAwareCache.DEFAULT_QUALITY

    # ── 线程安全测试 ──

    def test_thread_safety(self, cache):
        """并发写入不应崩溃"""
        import threading

        def writer(key, n):
            for _ in range(n):
                cache.record_outcome(key, True)

        threads = [threading.Thread(target=writer, args=(f"key_{i}", 20)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i in range(5):
            assert cache.get_quality(f"key_{i}") > OutcomeAwareCache.DEFAULT_QUALITY


class TestGlobalInstance:
    """全局实例管理"""

    def test_singleton(self):
        """get_outcome_cache 应返回单例"""
        with tempfile.TemporaryDirectory() as d:
            # 清理全局实例
            import task_router.outcome_cache as _oc
            _oc._quality_cache = None

            c1 = get_outcome_cache(d)
            c2 = get_outcome_cache(d)
            assert c1 is c2

            # 清理
            _oc._quality_cache = None


class TestOATSPattern:
    """OATS 模式验证 — 结果感知的缓存质量演化"""

    @pytest.fixture
    def cache_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_quality_convergence_pattern(self, cache_dir):
        """
        OATS 核心思想验证：
        成功的缓存条目应逐渐获得更高的质量分，
        失败的条目应逐渐降低。
        """
        cache = OutcomeAwareCache(cache_dir)

        # 模拟：某些查询模式总是成功，某些总是失败
        good_queries = ["q1", "q2", "q3"]
        bad_queries = ["q4", "q5", "q6"]

        for _ in range(30):
            for q in good_queries:
                cache.record_outcome(q, True)
            for q in bad_queries:
                cache.record_outcome(q, False)

        # 好查询的阈值应低于坏查询
        for good_q in good_queries:
            for bad_q in bad_queries:
                g_threshold = cache.get_effective_threshold(good_q, 0.85)
                b_threshold = cache.get_effective_threshold(bad_q, 0.85)
                assert g_threshold < b_threshold, \
                    f"成功查询 {good_q} 的阈值应低于失败查询 {bad_q}"

    def test_gradual_adaptation(self, cache_dir):
        """质量分应逐步适应（非突变）"""
        cache = OutcomeAwareCache(cache_dir)

        # 初始成功
        for _ in range(20):
            cache.record_outcome("key", True)
        q_before = cache.get_quality("key")

        # 开始失败
        cache.record_outcome("key", False)
        q_after = cache.get_quality("key")

        # 变化应很小（EMA 平滑）
        change = abs(q_after - q_before)
        assert change < 0.1, f"质量分变化应平滑: {change}"
