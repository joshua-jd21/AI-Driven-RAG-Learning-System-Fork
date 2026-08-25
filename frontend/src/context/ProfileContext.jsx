import React, { createContext, useContext, useEffect, useState } from 'react';

const ProfileContext = createContext();

export const useProfile = () => useContext(ProfileContext);

const LEARNER_ID_KEY = 'learnos_learner_id';

const DEFAULT_PROFILE = {
  learner_id: '',
  name: '',
  academic_level: 'class_11',
  grade: '11',
  board: 'CBSE',
  language: 'English',
  profile_version: 1,
  exam_target: ['JEE'],
  learning_style: 'visual',
  pace_preference: 'balanced',
  weak_subjects: [],
  confidence_map: {
    Chemistry: 50,
    Physics: 50,
    Mathematics: 50,
  },
  created_at: '',
  updated_at: '',
};

const API_HEADERS = {
  'Content-Type': 'application/json',
};

const ensureLearnerId = () => {
  let learnerId = localStorage.getItem(LEARNER_ID_KEY);
  if (!learnerId) {
    learnerId = `user-${Math.random().toString(36).slice(2, 11)}`;
    localStorage.setItem(LEARNER_ID_KEY, learnerId);
  }
  return learnerId;
};

const normalizeProfile = (raw, fallbackId = '') => {
  const source = raw && typeof raw === 'object' ? raw : {};
  const confidenceMap = source.confidence_map && typeof source.confidence_map === 'object'
    ? source.confidence_map
    : { ...DEFAULT_PROFILE.confidence_map };

  return {
    ...DEFAULT_PROFILE,
    ...source,
    learner_id: source.learner_id || fallbackId || '',
    exam_target: Array.isArray(source.exam_target) ? source.exam_target : [...DEFAULT_PROFILE.exam_target],
    weak_subjects: Array.isArray(source.weak_subjects) ? source.weak_subjects : [],
    confidence_map: confidenceMap,
    grade: source.grade || '',
    board: source.board || '',
    language: source.language || 'English',
    profile_version: Number.isFinite(Number(source.profile_version)) ? Number(source.profile_version) : 1,
  };
};

const buildGuestProfile = (learnerId) => ({
  ...DEFAULT_PROFILE,
  learner_id: learnerId,
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
});

export const ProfileProvider = ({ children }) => {
  const [profile, setProfile] = useState(DEFAULT_PROFILE);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function loadProfile() {
      const learnerId = ensureLearnerId();
      const baseFallback = buildGuestProfile(learnerId);

      try {
        const response = await fetch(`/api/profile/${encodeURIComponent(learnerId)}`);
        if (response.ok) {
          const data = await response.json();
          if (!cancelled) {
            setProfile(normalizeProfile(data, learnerId));
          }
          return;
        }

        if (response.status === 404) {
          const createResponse = await fetch('/api/profile', {
            method: 'POST',
            headers: API_HEADERS,
            body: JSON.stringify(baseFallback),
          });
          if (createResponse.ok) {
            const data = await createResponse.json();
            const savedProfile = normalizeProfile(data.profile || data, learnerId);
            if (!cancelled) {
              setProfile(savedProfile);
            }
            return;
          }
        }
      } catch (err) {
        console.warn('Failed to load profile from server, using local fallback:', err);
      }

      if (!cancelled) {
        setProfile(baseFallback);
      }
    }

    loadProfile().finally(() => {
      if (!cancelled) {
        setLoading(false);
      }
    });

    return () => {
      cancelled = true;
    };
  }, []);

  const updateProfile = async (updates) => {
    const learnerId = profile.learner_id || ensureLearnerId();
    const draftProfile = normalizeProfile(
      {
        ...profile,
        ...updates,
        learner_id: learnerId,
        updated_at: new Date().toISOString(),
      },
      learnerId,
    );

    try {
      const response = await fetch('/api/profile', {
        method: 'POST',
        headers: API_HEADERS,
        body: JSON.stringify(draftProfile),
      });

      if (response.ok) {
        const data = await response.json();
        const savedProfile = normalizeProfile(data.profile || data, learnerId);
        setProfile(savedProfile);
        localStorage.setItem(LEARNER_ID_KEY, savedProfile.learner_id);
        return savedProfile;
      }
    } catch (err) {
      console.warn('Profile save failed; keeping local draft until the next sync:', err);
    }

    setProfile(draftProfile);
    localStorage.setItem(LEARNER_ID_KEY, learnerId);
    return draftProfile;
  };

  const resetProfile = async () => {
    const newId = `user-${Math.random().toString(36).slice(2, 11)}`;
    localStorage.setItem(LEARNER_ID_KEY, newId);
    localStorage.removeItem('learnos_profile');

    const freshProfile = buildGuestProfile(newId);
    try {
      const response = await fetch('/api/profile', {
        method: 'POST',
        headers: API_HEADERS,
        body: JSON.stringify(freshProfile),
      });

      if (response.ok) {
        const data = await response.json();
        const savedProfile = normalizeProfile(data.profile || data, newId);
        setProfile(savedProfile);
        return savedProfile;
      }
    } catch (err) {
      console.warn('Profile reset failed; using local fallback profile:', err);
    }

    setProfile(freshProfile);

    try {
      await fetch('/api/persist', {
        method: 'POST',
        headers: API_HEADERS,
        body: JSON.stringify({ filename: 'history.json', payload: { sessions: [] } }),
      });

      await fetch('/api/persist', {
        method: 'POST',
        headers: API_HEADERS,
        body: JSON.stringify({
          filename: 'analytics.json',
          payload: {
            total_sessions: 0,
            total_watch_time_seconds: 0,
            topics_covered: [],
            weak_topic_flags: [],
            daily_activity: [],
            subject_distribution: {},
          },
        }),
      });
    } catch (err) {
      console.warn('Failed to clear server-side analytics/history during reset:', err);
    }

    return freshProfile;
  };

  return (
    <ProfileContext.Provider value={{ profile, updateProfile, resetProfile, loading }}>
      {children}
    </ProfileContext.Provider>
  );
};
